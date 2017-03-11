using System;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents convolution neural layer as a list of convolution neurons that produce feature maps
  /// </summary>
  public class ConvolutionalLayer : LayerNode<double[,], FeatureMap>
  {
    #region Fields

    private PoolingMap m_PoolingMap;
    private int m_InputSize;
    private int m_WindowSize;
    private int m_OutputSize;
    private int m_Stride;

    #endregion

    #region .ctor

    public ConvolutionalLayer(int inputDim,
                              int inputSize,
                              int windowSize,
                              int stride=0)
      : base(inputDim)
    {
      if (inputSize <= 0 || windowSize <= 0)
        throw new MLException("ConvolutionNeuron.ctor(inputSize<=0|windowSize<=0)");
      if (windowSize > inputSize)
        throw new MLException("ConvolutionNeuron.ctor(windowSize>inputSize)");
      if (stride < 0)
        throw new MLException("ConvolutionNeuron.ctor(stride<0)");

      m_InputSize  = inputSize;
      m_WindowSize = windowSize;
      m_Stride     = (stride == 0) ? m_WindowSize/2 : stride;
    }

    #endregion

    #region Properties

    /// <summary>
    /// Total count of neurons
    /// </summary>
    public int FeatureMapCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Size of square input matrix
    /// </summary>
    public int InputSize { get { return m_InputSize; } }

    /// <summary>
    /// Size of square conwolution window
    /// </summary>
    public int WindowSize { get { return m_WindowSize; } }

    /// <summary>
    /// Size of square output matrix
    /// </summary>
    public int OutputSize { get { return m_OutputSize; } }

    /// <summary>
    /// An overlapping step during convolution calculation process.
    /// 1 leads to maximum overlapping between neighour kernel windows
    /// 0 defaults to (int)m_WindowSize/2 (minimum overlapping that covers center)
    /// </summary>
    public int Stride { get { return m_Stride; } }

    /// <summary>
    /// Pooling for this layer (null means no pooling)
    /// </summary>
    public PoolingMap PoolingMap
    {
      get { return m_PoolingMap; }
      set { m_PoolingMap = value; }
    }

    #endregion

    #region Public

    /// <summary>
    /// Add existing neuron in the end of the layer
    /// </summary>
    public override void AddNeuron(FeatureMap neuron)
    {
      if (neuron==null)
        throw new MLException("Neuron can not be null");
      if (neuron.InputDim != this.InputDim)
        throw new MLException("Neuron input dimension differs with layer's one");

      this.AddSubNode(neuron);
    }

    /// <summary>
    /// Randomizes layer's neuron weights
    /// </summary>
    public void Randomize(int seed=0)
    {
      foreach (var neuron in this.SubNodes)
        neuron.RandomizeParameters(seed);
    }

    /// <summary>
    /// Calculates result array produced by layer
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[][,] Calculate(double[][,] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      var result = base.Calculate(input);
      if (m_PoolingMap != null)
      {
        var len = result.Length;
        for (int k=0; k<len; k++)
          result[k] = m_PoolingMap.Calculate(result[k]);
      }

      return result;
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      m_OutputSize = (m_PoolingMap != null) ?
                     m_PoolingMap.OutputSize :
                     (m_InputSize-m_WindowSize)/m_Stride+1;

      base.DoBuild();
    }

    #endregion
  }
}
