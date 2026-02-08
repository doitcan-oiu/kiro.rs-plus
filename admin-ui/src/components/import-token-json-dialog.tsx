import { useState, useCallback, useRef } from 'react'
import { toast } from 'sonner'
import { Upload, FileJson, CheckCircle2, XCircle, AlertCircle, Loader2 } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { useImportTokenJson, useDeleteCredential } from '@/hooks/use-credentials'
import { getCredentialBalance, setCredentialDisabled } from '@/api/credentials'
import { extractErrorMessage } from '@/lib/utils'
import type { TokenJsonItem, ImportItemResult, ImportSummary } from '@/types/api'

interface ImportTokenJsonDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

type Step = 'input' | 'preview' | 'result' | 'verifying'

// 验活结果
interface VerifyItemResult {
  index: number
  credentialId?: number
  status: 'pending' | 'verifying' | 'verified' | 'failed' | 'skipped' | 'rolled_back' | 'rollback_failed'
  usage?: string
  error?: string
  rollbackError?: string
}

export function ImportTokenJsonDialog({ open, onOpenChange }: ImportTokenJsonDialogProps) {
  const [step, setStep] = useState<Step>('input')
  const [jsonText, setJsonText] = useState('')
  const [parsedItems, setParsedItems] = useState<TokenJsonItem[]>([])
  const [previewResults, setPreviewResults] = useState<ImportItemResult[]>([])
  const [previewSummary, setPreviewSummary] = useState<ImportSummary | null>(null)
  const [finalResults, setFinalResults] = useState<ImportItemResult[]>([])
  const [finalSummary, setFinalSummary] = useState<ImportSummary | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [enableVerify, setEnableVerify] = useState(false)
  const [verifyResults, setVerifyResults] = useState<VerifyItemResult[]>([])
  const [verifyProgress, setVerifyProgress] = useState({ current: 0, total: 0 })
  const [isVerifying, setIsVerifying] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { mutate: importMutate, isPending } = useImportTokenJson()
  const { mutateAsync: deleteCredential } = useDeleteCredential()

  const resetState = useCallback(() => {
    setStep('input')
    setJsonText('')
    setParsedItems([])
    setPreviewResults([])
    setPreviewSummary(null)
    setFinalResults([])
    setFinalSummary(null)
    setEnableVerify(false)
    setVerifyResults([])
    setVerifyProgress({ current: 0, total: 0 })
    setIsVerifying(false)
  }, [])

  const handleClose = useCallback(() => {
    if (isVerifying) return // 验活中不允许关闭
    onOpenChange(false)
    setTimeout(resetState, 200)
  }, [onOpenChange, resetState, isVerifying])

  // 解析 JSON
  const parseJson = useCallback((text: string): TokenJsonItem[] | null => {
    try {
      const parsed = JSON.parse(text)
      const items = Array.isArray(parsed) ? parsed : [parsed]
      const validItems = items.filter(
        (item) => item && typeof item === 'object' && item.refreshToken
      )
      if (validItems.length === 0) {
        toast.error('JSON 中没有找到有效的凭据（需要包含 refreshToken 字段）')
        return null
      }
      return validItems
    } catch {
      toast.error('JSON 格式无效')
      return null
    }
  }, [])

  // 文件拖放
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (!file) return
    if (!file.name.endsWith('.json')) {
      toast.error('请上传 JSON 文件')
      return
    }
    const reader = new FileReader()
    reader.onload = (event) => setJsonText(event.target?.result as string)
    reader.readAsText(file)
  }, [])

  // 文件选择
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (event) => setJsonText(event.target?.result as string)
    reader.readAsText(file)
  }, [])

  // 预览（dry-run）
  const handlePreview = useCallback(() => {
    const items = parseJson(jsonText)
    if (!items) return
    setParsedItems(items)
    importMutate(
      { dryRun: true, items },
      {
        onSuccess: (response) => {
          setPreviewResults(response.items)
          setPreviewSummary(response.summary)
          setStep('preview')
        },
        onError: (error) => {
          toast.error(`预览失败: ${extractErrorMessage(error)}`)
        },
      }
    )
  }, [jsonText, parseJson, importMutate])

  // 回滚凭据（禁用 + 删除）
  const rollbackCredential = async (id: number): Promise<{ success: boolean; error?: string }> => {
    try {
      await setCredentialDisabled(id, true)
    } catch (error) {
      return { success: false, error: `禁用失败: ${extractErrorMessage(error)}` }
    }
    try {
      await deleteCredential(id)
      return { success: true }
    } catch (error) {
      return { success: false, error: `删除失败: ${extractErrorMessage(error)}` }
    }
  }

  // 验活流程
  const runVerification = useCallback(async (results: ImportItemResult[]) => {
    const addedItems = results.filter(r => r.action === 'added' && r.credentialId)
    if (addedItems.length === 0) {
      toast.info('没有新增凭据需要验活')
      return
    }

    setIsVerifying(true)
    setStep('verifying')
    setVerifyProgress({ current: 0, total: addedItems.length })

    const initialVerifyResults: VerifyItemResult[] = addedItems.map(item => ({
      index: item.index,
      credentialId: item.credentialId,
      status: 'pending',
    }))
    setVerifyResults(initialVerifyResults)

    let successCount = 0
    let failCount = 0

    for (let i = 0; i < addedItems.length; i++) {
      const item = addedItems[i]
      const credId = item.credentialId!

      // 更新为验活中
      setVerifyResults(prev => prev.map((r, idx) =>
        idx === i ? { ...r, status: 'verifying' } : r
      ))

      try {
        await new Promise(resolve => setTimeout(resolve, 1000))
        const balance = await getCredentialBalance(credId)
        successCount++
        setVerifyResults(prev => prev.map((r, idx) =>
          idx === i ? { ...r, status: 'verified', usage: `${balance.currentUsage}/${balance.usageLimit}` } : r
        ))
      } catch (error) {
        failCount++
        // 验活失败，回滚
        const rollback = await rollbackCredential(credId)
        setVerifyResults(prev => prev.map((r, idx) =>
          idx === i ? {
            ...r,
            status: rollback.success ? 'rolled_back' : 'rollback_failed',
            error: extractErrorMessage(error),
            rollbackError: rollback.error,
          } : r
        ))
      }

      setVerifyProgress({ current: i + 1, total: addedItems.length })
    }

    setIsVerifying(false)

    if (failCount === 0) {
      toast.success(`全部 ${successCount} 个凭据验活成功`)
    } else {
      toast.info(`验活完成：成功 ${successCount}，失败 ${failCount}`)
    }
  }, [deleteCredential])

  // 确认导入
  const handleConfirmImport = useCallback(() => {
    importMutate(
      { dryRun: false, items: parsedItems },
      {
        onSuccess: (response) => {
          setFinalResults(response.items)
          setFinalSummary(response.summary)

          if (enableVerify) {
            // 开启验活模式：导入后自动验活
            if (response.summary.added > 0) {
              toast.success(`成功导入 ${response.summary.added} 个凭据，开始验活...`)
            }
            runVerification(response.items)
          } else {
            // 普通模式：直接显示结果
            setStep('result')
            if (response.summary.added > 0) {
              toast.success(`成功导入 ${response.summary.added} 个凭据`)
            }
          }
        },
        onError: (error) => {
          toast.error(`导入失败: ${extractErrorMessage(error)}`)
        },
      }
    )
  }, [parsedItems, importMutate, enableVerify, runVerification])

  // 渲染图标
  const renderActionIcon = (action: string) => {
    switch (action) {
      case 'added': return <CheckCircle2 className="h-4 w-4 text-green-500" />
      case 'skipped': return <AlertCircle className="h-4 w-4 text-yellow-500" />
      case 'invalid': return <XCircle className="h-4 w-4 text-red-500" />
      default: return null
    }
  }

  const renderActionText = (action: string) => {
    switch (action) {
      case 'added': return <span className="text-green-600">添加</span>
      case 'skipped': return <span className="text-yellow-600">跳过</span>
      case 'invalid': return <span className="text-red-600">无效</span>
      default: return action
    }
  }

  const getVerifyStatusIcon = (status: VerifyItemResult['status']) => {
    switch (status) {
      case 'pending': return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
      case 'verifying': return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
      case 'verified': return <CheckCircle2 className="w-5 h-5 text-green-500" />
      case 'failed':
      case 'rollback_failed': return <XCircle className="w-5 h-5 text-red-500" />
      case 'rolled_back': return <AlertCircle className="w-5 h-5 text-yellow-500" />
      case 'skipped': return <AlertCircle className="w-5 h-5 text-gray-400" />
    }
  }

  const getVerifyStatusText = (result: VerifyItemResult) => {
    switch (result.status) {
      case 'pending': return '等待中'
      case 'verifying': return '验活中...'
      case 'verified': return '验活成功'
      case 'failed': return '验活失败'
      case 'rolled_back': return '验活失败（已排除）'
      case 'rollback_failed': return '验活失败（未排除）'
      case 'skipped': return '跳过'
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileJson className="h-5 w-5" />
            导入凭据
          </DialogTitle>
          <DialogDescription>
            {step === 'input' && '粘贴或上传 JSON 文件以批量导入凭据'}
            {step === 'preview' && '预览导入结果，确认后执行导入'}
            {step === 'result' && '导入完成'}
            {step === 'verifying' && '正在验活导入的凭据...'}
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto py-4">
          {/* Step 1: Input */}
          {step === 'input' && (
            <div className="space-y-4">
              {/* 拖放区域 */}
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  isDragging
                    ? 'border-primary bg-primary/5'
                    : 'border-muted-foreground/25 hover:border-muted-foreground/50'
                }`}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="h-10 w-10 mx-auto mb-4 text-muted-foreground" />
                <p className="text-sm text-muted-foreground mb-2">
                  拖放 JSON 文件到此处，或点击选择文件
                </p>
                <p className="text-xs text-muted-foreground">
                  支持单个凭据或凭据数组格式
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </div>

              {/* 分隔线 */}
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">或</span>
                </div>
              </div>

              {/* 文本输入 */}
              <div className="space-y-2">
                <label className="text-sm font-medium">直接粘贴 JSON</label>
                <textarea
                  className="w-full h-48 p-3 text-sm font-mono border rounded-md bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder='{"refreshToken": "...", "provider": "BuilderId", ...}'
                  value={jsonText}
                  onChange={(e) => setJsonText(e.target.value)}
                />
              </div>
            </div>
          )}

          {/* Step 2: Preview */}
          {step === 'preview' && previewSummary && (
            <div className="space-y-4">
              {/* 统计 */}
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center p-3 bg-muted rounded-lg">
                  <div className="text-2xl font-bold">{previewSummary.parsed}</div>
                  <div className="text-xs text-muted-foreground">解析</div>
                </div>
                <div className="text-center p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{previewSummary.added}</div>
                  <div className="text-xs text-muted-foreground">将添加</div>
                </div>
                <div className="text-center p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-600">{previewSummary.skipped}</div>
                  <div className="text-xs text-muted-foreground">跳过</div>
                </div>
                <div className="text-center p-3 bg-red-50 dark:bg-red-950 rounded-lg">
                  <div className="text-2xl font-bold text-red-600">{previewSummary.invalid}</div>
                  <div className="text-xs text-muted-foreground">无效</div>
                </div>
              </div>

              {/* 预览列表 */}
              <div className="border rounded-lg overflow-hidden">
                <div className="max-h-48 overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-muted sticky top-0">
                      <tr>
                        <th className="text-left p-2 font-medium">#</th>
                        <th className="text-left p-2 font-medium">指纹</th>
                        <th className="text-left p-2 font-medium">状态</th>
                        <th className="text-left p-2 font-medium">原因</th>
                      </tr>
                    </thead>
                    <tbody>
                      {previewResults.map((item) => (
                        <tr key={item.index} className="border-t">
                          <td className="p-2">{item.index + 1}</td>
                          <td className="p-2 font-mono text-xs">{item.fingerprint}</td>
                          <td className="p-2">
                            <div className="flex items-center gap-1">
                              {renderActionIcon(item.action)}
                              {renderActionText(item.action)}
                            </div>
                          </td>
                          <td className="p-2 text-muted-foreground text-xs">
                            {item.reason || '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 验活开关 */}
              {previewSummary.added > 0 && (
                <div className="flex items-center justify-between p-3 border rounded-lg bg-muted/50">
                  <div>
                    <div className="text-sm font-medium">导入后自动验活</div>
                    <div className="text-xs text-muted-foreground">
                      逐个检查凭据有效性，失败的自动排除
                    </div>
                  </div>
                  <Switch checked={enableVerify} onCheckedChange={setEnableVerify} />
                </div>
              )}
            </div>
          )}

          {/* Step 3: Result (普通模式) */}
          {step === 'result' && finalSummary && (
            <div className="space-y-4">
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center p-3 bg-muted rounded-lg">
                  <div className="text-2xl font-bold">{finalSummary.parsed}</div>
                  <div className="text-xs text-muted-foreground">解析</div>
                </div>
                <div className="text-center p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{finalSummary.added}</div>
                  <div className="text-xs text-muted-foreground">已添加</div>
                </div>
                <div className="text-center p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-600">{finalSummary.skipped}</div>
                  <div className="text-xs text-muted-foreground">跳过</div>
                </div>
                <div className="text-center p-3 bg-red-50 dark:bg-red-950 rounded-lg">
                  <div className="text-2xl font-bold text-red-600">{finalSummary.invalid}</div>
                  <div className="text-xs text-muted-foreground">无效</div>
                </div>
              </div>

              <div className="border rounded-lg overflow-hidden">
                <div className="max-h-64 overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-muted sticky top-0">
                      <tr>
                        <th className="text-left p-2 font-medium">#</th>
                        <th className="text-left p-2 font-medium">指纹</th>
                        <th className="text-left p-2 font-medium">状态</th>
                        <th className="text-left p-2 font-medium">凭据 ID</th>
                      </tr>
                    </thead>
                    <tbody>
                      {finalResults.map((item) => (
                        <tr key={item.index} className="border-t">
                          <td className="p-2">{item.index + 1}</td>
                          <td className="p-2 font-mono text-xs">{item.fingerprint}</td>
                          <td className="p-2">
                            <div className="flex items-center gap-1">
                              {renderActionIcon(item.action)}
                              {renderActionText(item.action)}
                            </div>
                          </td>
                          <td className="p-2">
                            {item.credentialId ? `#${item.credentialId}` : item.reason || '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Step: Verifying (验活模式) */}
          {step === 'verifying' && (
            <div className="space-y-4">
              {/* 进度条 */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>{isVerifying ? '验活进度' : '验活完成'}</span>
                  <span>{verifyProgress.current} / {verifyProgress.total}</span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all"
                    style={{ width: verifyProgress.total > 0 ? `${(verifyProgress.current / verifyProgress.total) * 100}%` : '0%' }}
                  />
                </div>
              </div>

              {/* 统计 */}
              <div className="flex gap-4 text-sm">
                <span className="text-green-600 dark:text-green-400">
                  ✓ 成功: {verifyResults.filter(r => r.status === 'verified').length}
                </span>
                <span className="text-yellow-600 dark:text-yellow-400">
                  ⚠ 已排除: {verifyResults.filter(r => r.status === 'rolled_back').length}
                </span>
                <span className="text-red-600 dark:text-red-400">
                  ✗ 失败: {verifyResults.filter(r => r.status === 'rollback_failed').length}
                </span>
              </div>

              {/* 结果列表 */}
              <div className="border rounded-md divide-y max-h-[300px] overflow-y-auto">
                {verifyResults.map((result) => (
                  <div key={result.index} className="p-3">
                    <div className="flex items-start gap-3">
                      {getVerifyStatusIcon(result.status)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">
                            凭据 #{result.credentialId || result.index + 1}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {getVerifyStatusText(result)}
                          </span>
                        </div>
                        {result.usage && (
                          <div className="text-xs text-muted-foreground mt-1">
                            用量: {result.usage}
                          </div>
                        )}
                        {result.error && (
                          <div className="text-xs text-red-600 dark:text-red-400 mt-1">
                            {result.error}
                          </div>
                        )}
                        {result.rollbackError && (
                          <div className="text-xs text-red-600 dark:text-red-400 mt-1">
                            回滚失败: {result.rollbackError}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          {step === 'input' && (
            <>
              <Button variant="outline" onClick={handleClose}>
                取消
              </Button>
              <Button onClick={handlePreview} disabled={!jsonText.trim() || isPending}>
                {isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    解析中...
                  </>
                ) : (
                  '预览'
                )}
              </Button>
            </>
          )}

          {step === 'preview' && (
            <>
              <Button variant="outline" onClick={() => setStep('input')}>
                返回
              </Button>
              <Button
                onClick={handleConfirmImport}
                disabled={isPending || (previewSummary?.added ?? 0) === 0}
              >
                {isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    导入中...
                  </>
                ) : enableVerify ? (
                  `导入并验活 (${previewSummary?.added ?? 0})`
                ) : (
                  `确认导入 (${previewSummary?.added ?? 0})`
                )}
              </Button>
            </>
          )}

          {step === 'result' && (
            <Button onClick={handleClose}>完成</Button>
          )}

          {step === 'verifying' && (
            <Button onClick={handleClose} disabled={isVerifying}>
              {isVerifying ? '验活中...' : '完成'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
