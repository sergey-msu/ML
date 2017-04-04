using System;
using ML.Core.ComputingNetworks;
using ML.Core;
using ML.Contracts;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents "deep" layer - a building block of deep networks (i.e. CNN)
  /// that accepts 3D input (a series of 2D channels) and returns 3D output (i.e. series of feature maps in CNN)
  /// </summary>
  public abstract class DeepLayerBase : ComputingNode<double[,,], double[,,]>
  {
    #region Fields

    protected bool m_IsTraining;
    protected IActivationFunction m_ActivationFunction;

    internal  int m_InputSize;
    internal  int m_InputDepth;
    protected int m_OutputSize;
    protected int m_OutputDepth;
    protected int m_WindowSize;
    protected int m_Stride;
    protected int m_Padding;

    protected double[,,] m_Value;
    protected double[,,] m_Error;

    #endregion

    #region .ctor

    public DeepLayerBase(int outputDepth,
                         int windowSize,
                         int stride,
                         int padding=0)
    {
      if (outputDepth <= 0)
        throw new MLException("DeepLayerBase.ctor(outputDepth<=0)");
      if (windowSize <= 0)
        throw new MLException("DeepLayerBase.ctor(windowSize<=0)");
      if (stride <= 0)
        throw new MLException("DeepLayerBase.ctor(stride<=0)");
      if (padding < 0)
        throw new MLException("DeepLayerBase.ctor(padding<0)");

      m_WindowSize  = windowSize;
      m_Stride      = stride;
      m_Padding     = padding;
      m_OutputDepth = outputDepth;
    }

    #endregion

    #region Properties

    /// <summary>
    /// If true, indicates that layer is in training mode,
    /// so it can store some additional values (i.e. errors, net values, derivatives etc.)
    /// </summary>
    public bool IsTraining
    {
      get { return m_IsTraining; }
      set { m_IsTraining=value; }
    }

    /// <summary>
    /// Count of input channels
    /// </summary>
    public int InputDepth { get { return m_InputDepth; } }

    /// <summary>
    /// Size of squared input channel
    /// </summary>
    public int InputSize { get { return m_InputSize; } }

    /// <summary>
    /// Count of output feature map channels
    /// </summary>
    public int OutputDepth { get { return m_OutputDepth; } }

    /// <summary>
    /// Size of square output feature map channel
    /// </summary>
    public int OutputSize { get { return m_OutputSize; } }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Size of square convolution window
    /// </summary>
    public int WindowSize { get { return m_WindowSize; } }

    /// <summary>
    /// An overlapping step during convolution calculation process.
    /// 1 leads to maximum overlapping between neighour kernel windows
    /// </summary>
    public int Stride { get { return m_Stride; } }

    /// <summary>
    /// Padding of the input channel
    /// </summary>
    public int Padding { get { return m_Padding; } }

    /// <summary>
    /// Calculated value
    /// </summary>
    public double[,,] Value { get { return m_Value; } }

    /// <summary>
    /// Saved error value
    /// </summary>
    public double[,,] Error { get { return m_Error; } }

    #endregion

    /// <summary>
    /// Calculates net value's derivative
    /// </summary>
    public double Derivative(int p, int i, int j)
    {
      return m_ActivationFunction.DerivativeFromValue(m_Value[p, i, j]);
    }

    public override double[,,] Calculate(double[,,] input)
    {
      if (m_InputDepth != input.GetLength(0))
        throw new MLException("Incorrect input depth");
      if (m_InputSize != input.GetLength(1))
        throw new MLException("Incorrect input size");
      if (m_InputSize != input.GetLength(2))
        throw new MLException("Incorrect input size");

      return DoCalculate(input);
    }

    public override void DoBuild()
    {
      base.DoBuild();

      m_OutputSize = (m_InputSize - m_WindowSize + 2*m_Padding)/m_Stride + 1;
      if (m_OutputSize <= 0)
        throw new MLException("Output tensor is empty. Check input shape datas");

      m_Value = new double[m_OutputDepth, m_OutputSize, m_OutputSize];

      if (m_IsTraining)
      {
        m_Error = new double[m_OutputDepth, m_OutputSize, m_OutputSize];
      }
    }

    /// <summary>
    /// Randomizes layer parameters (i.e. kernel weights, biases etc.)
    /// </summary>
    public virtual void RandomizeParameters(int seed)
    {
    }

    protected abstract double[,,] DoCalculate(double[,,] input);
  }
}
