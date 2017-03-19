using System;
using System.Collections.Generic;
using ML.Core.ComputingNetworks;
using ML.Core;
using ML.Contracts;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents "deep" layer - a building block of deep networks (i.e. CNN)
  /// that accepts 3D input (a series of 2D channels) and returns 3D output (i.e. series of feature maps in CNN)
  /// </summary>
  public abstract class DeepLayerBase : ComputingNode<double[,,], double[,,]>
  {
    #region Fields

    private IActivationFunction m_ActivationFunction;

    protected readonly bool m_IsTraining;

    protected readonly int  m_WindowSize;
    protected readonly int  m_Stride;
    protected readonly int  m_Padding;
    protected readonly int  m_InputDepth;
    protected readonly int  m_InputSize;
    protected readonly int  m_OutputDepth;
    protected readonly int  m_OutputSize;

    protected readonly double[,,] m_Value;
    protected readonly double[,,] m_Error;

    #endregion

    #region .ctor

    public DeepLayerBase(int inputDepth,
                         int inputSize,
                         int outputDepth,
                         int outputSize,
                         int windowSize,
                         int stride,
                         int padding=0,
                         bool isTraining=false)
    {
      if (inputDepth <= 0)
        throw new MLException("DeepLayerBase.ctor(inputDepth<=0)");
      if (inputSize <= 0)
        throw new MLException("DeepLayerBase.ctor(inputSize<=0)");
      if (outputDepth <= 0)
        throw new MLException("DeepLayerBase.ctor(outputDepth<=0)");
      if (outputSize <= 0)
        throw new MLException("DeepLayerBase.ctor(outputSize<=0)");
      if (windowSize <= 0)
        throw new MLException("ConvolutionalLayer.ctor(windowSize<=0)");
      if (windowSize > inputSize)
        throw new MLException("ConvolutionalLayer.ctor(windowSize>inputSize)");
      if (stride <= 0)
        throw new MLException("ConvolutionalLayer.ctor(stride<=0)");
      if (padding < 0)
        throw new MLException("ConvolutionalLayer.ctor(padding<0)");

      m_IsTraining = isTraining;

      m_WindowSize  = windowSize;
      m_Stride      = stride;
      m_Padding     = padding;

      m_InputDepth  = inputDepth;
      m_InputSize   = inputSize;
      m_OutputDepth = outputDepth;
      m_OutputSize  = outputSize;

      m_Value = new double[m_OutputDepth, m_OutputSize, m_OutputSize];

      if (isTraining)
      {
        m_Error = new double[m_OutputDepth, m_OutputSize, m_OutputSize];
      }
    }

    #endregion

    #region Properties

    /// <summary>
    /// If true, indicates that layer is not in production mode,
    /// so it can store some additional values (i.e. errors, net values, derivatives etc.)
    /// </summary>
    public bool IsTraining { get { return m_IsTraining; } }

    /// <summary>
    /// Count of input channels
    /// </summary>
    public int InputDepth { get { return m_InputDepth; } }

    /// <summary>
    /// Size of square input channel
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
    /// Randomizes layer parameters (i.e. kernel weights, biases etc.)
    /// </summary>
    public abstract void RandomizeParameters(int seed);

    /// <summary>
    /// Calculates value's derivative
    /// </summary>
    public double Derivative(int p, int i, int j)
    {
      return ActivationFunction.DerivativeFromValue(Value[p, i, j]);
    }
  }
}
