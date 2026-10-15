using Enums;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.EP.WebGpu;

namespace SmartData.Lib.Services.Base
{
    public abstract class BaseAIConsumer<TInput, TOutput> : CancellableServiceBase
        where TInput : class
        where TOutput : class, new()
    {
        protected InferenceSession _session;

        private static readonly OnnxRuntimeProvider[] _executionProviders =
        [
            OnnxRuntimeProvider.CUDA,
            OnnxRuntimeProvider.WebGPU,
            OnnxRuntimeProvider.ROCm,
            OnnxRuntimeProvider.DirectML
        ];

        private static readonly object _webGpuLock = new object();
        private static volatile bool _webGpuRegistered = false;
        private static readonly Dictionary<string, string> _emptyProviderOptions = [];

        private readonly SemaphoreSlim _loadModelSemaphore = new SemaphoreSlim(1, 1);

        public string ModelPath { get; set; }

        protected bool _isModelLoaded = false;
        public bool IsModelLoaded
        {
            get => _isModelLoaded;
            private set
            {
                _isModelLoaded = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseAIConsumer{TInput, TOutput}"/> class.
        /// </summary>
        /// <param name="modelPath">Path to the machine learning model.</param>
        protected BaseAIConsumer(string modelPath)
        {
            ModelPath = modelPath;
        }

        /// <summary>
        /// Retrieves an array of input column names for the machine learning model.
        /// </summary>
        /// <returns>An array of strings representing the input column names.</returns>
        protected abstract string[] GetInputColumns();

        /// <summary>
        /// Retrieves an array of output column names for the machine learning model.
        /// </summary>
        /// <returns>An array of strings representing the output column names.</returns>
        protected abstract string[] GetOutputColumns();

        /// <summary>
        /// Loads the machine learning model and initializes the prediction session.
        /// </summary>
        protected virtual async Task LoadModelAsync()
        {
            await _loadModelSemaphore.WaitAsync().ConfigureAwait(false);
            try
            {
                if (IsModelLoaded)
                {
                    return;
                }

                ResetState();

                bool sessionCreated = false;
                foreach (OnnxRuntimeProvider provider in _executionProviders)
                {
                    SessionOptions sessionOptions = CreateSessionOptions();
                    if (TryAppendProvider(sessionOptions, provider))
                    {
                        try
                        {
                            sessionOptions.ApplyConfiguration();
                            _session = await Task.Run(() => new InferenceSession(ModelPath, sessionOptions)).ConfigureAwait(false);
                            sessionCreated = true;
                            break;
                        }
                        catch (Exception)
                        {
                            sessionOptions.Dispose();
                        }
                    }
                    else
                    {
                        sessionOptions.Dispose();
                    }
                }

                if (!sessionCreated)
                {
                    SessionOptions cpuOptions = CreateSessionOptions();
                    try
                    {
                        cpuOptions.ApplyConfiguration();
                        _session = await Task.Run(() => new InferenceSession(ModelPath, cpuOptions)).ConfigureAwait(false);
                    }
                    catch
                    {
                        cpuOptions.Dispose();
                        throw;
                    }
                }

                IsModelLoaded = true;
            }
            catch (Exception exception)
            {
                ResetState();

                throw new InvalidOperationException($"Failed to load model from {ModelPath}.", exception);
            }
            finally
            {
                _loadModelSemaphore.Release();
            }
        }

        private static SessionOptions CreateSessionOptions()
        {
            return new SessionOptions()
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableMemoryPattern = false
            };
        }

        /// <summary>
        /// Attempts to append the specified execution provider to the session options.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="provider"></param>
        /// <returns></returns>
        private bool TryAppendProvider(SessionOptions options, OnnxRuntimeProvider provider)
        {
            try
            {
                switch (provider)
                {
                    case OnnxRuntimeProvider.CUDA:
                        options.AppendExecutionProvider_CUDA();
                        break;
                    case OnnxRuntimeProvider.WebGPU:
                        return TryAppendWebGpu(options);
                    case OnnxRuntimeProvider.ROCm:
                        options.AppendExecutionProvider_ROCm();
                        break;
                    case OnnxRuntimeProvider.DirectML:
                        options.AppendExecutionProvider_DML();
                        break;
                }
                return true;
            }
            catch (Exception)
            {
                // The specified provider is not available.
                return false;
            }
        }

        private static bool TryAppendWebGpu(SessionOptions options)
        {
            try
            {
                OrtEnv env = OrtEnv.Instance();
                string epName = WebGpuEp.GetEpName();

                if (!_webGpuRegistered)
                {
                    lock (_webGpuLock)
                    {
                        if (!_webGpuRegistered)
                        {
                            string libPath = WebGpuEp.GetLibraryPath();
                            if (!string.IsNullOrEmpty(libPath) && File.Exists(libPath))
                            {
                                var existingDevices = env.GetEpDevices();
                                bool alreadyRegistered = false;
                                for (int i = 0; i < existingDevices.Count; i++)
                                {
                                    if (string.Equals(existingDevices[i].EpName, epName, StringComparison.Ordinal))
                                    {
                                        alreadyRegistered = true;
                                        break;
                                    }
                                }

                                if (!alreadyRegistered)
                                {
                                    env.RegisterExecutionProviderLibrary("webgpu_ep", libPath);
                                }
                                _webGpuRegistered = true;
                            }
                        }
                    }
                }

                if (!_webGpuRegistered)
                {
                    return false;
                }

                var devices = env.GetEpDevices();
                OrtEpDevice? webGpuDevice = null;
                for (int i = 0; i < devices.Count; i++)
                {
                    if (string.Equals(devices[i].EpName, epName, StringComparison.Ordinal))
                    {
                        webGpuDevice = devices[i];
                        break;
                    }
                }

                if (webGpuDevice is null)
                {
                    return false;
                }

                options.AppendExecutionProvider(env, [webGpuDevice], _emptyProviderOptions);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Unloads the machine learning model and releases associated resources.
        /// </summary>
        /// <remarks>
        /// This method disposes of the session and releases any resources
        /// associated with it, setting the session to null.
        /// After calling this method, the model will no longer be loaded.
        /// </remarks>
        protected virtual void UnloadModel()
        {
            if (_loadModelSemaphore.Wait(1000))
            {
                try
                {
                    ResetState();
                }
                finally
                {
                    _loadModelSemaphore.Release();
                }
            }
        }

        /// <summary>
        /// Resets the state of the AI consumer, disposing of the session and setting the model loaded flag to false.
        /// </summary>
        private void ResetState()
        {
            _session?.Dispose();
            _session = null;
            IsModelLoaded = false;
        }
    }
}
