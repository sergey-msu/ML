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

    private IFunction m_ActivationFunction;

    protected readonly int m_InputDepth;
    protected readonly int m_InputSize;
    protected readonly int m_OutputDepth;
    protected readonly int m_OutputSize;

    protected readonly double[,,]  m_Value;

    #endregion

    #region .ctor

    public DeepLayerBase(int inputDepth, int inputSize, int outputDepth, int outputSize)
    {
      if (inputDepth <= 0)
        throw new MLException("DeepLayerBase.ctor(inputDepth<=0)");
      if (inputSize <= 0)
        throw new MLException("DeepLayerBase.ctor(inputSize<=0)");
      if (outputDepth <= 0)
        throw new MLException("DeepLayerBase.ctor(outputDepth<=0)");
      if (outputSize <= 0)
        throw new MLException("DeepLayerBase.ctor(outputSize<=0)");

      m_InputDepth  = inputDepth;
      m_InputSize   = inputSize;
      m_OutputDepth = outputDepth;
      m_OutputSize  = outputSize;

      m_Value  = new double[m_OutputDepth, m_OutputSize, m_OutputSize];
    }

    #endregion

    #region Properties

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
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Calculated value
    /// </summary>
    public double[,,] Value { get { return m_Value; } }

    #endregion

    /// <summary>
    /// Randomizes layer parameters (i.e. kernel weights, biases etc.)
    /// </summary>
    public abstract void RandomizeParameters(int seed);
  }
}
