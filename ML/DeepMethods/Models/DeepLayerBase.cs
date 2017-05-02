using System;
using ML.Core.ComputingNetworks;
using ML.Core;
using ML.Contracts;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents "deep" layer - a building block of deep networks (i.e. CNN)
  /// that accepts 3D input (an array of 2D channels) and returns 3D output (e.g. array of feature maps in CNN)
  /// </summary>
  public abstract class DeepLayerBase : ComputingNode<double[][,], double[][,]>
  {
    #region Fields

    protected bool m_IsTraining;
    protected IActivationFunction m_ActivationFunction;

    protected int m_InputDepth;
    protected int m_InputHeight;
    protected int m_InputWidth;

    protected int m_OutputDepth;
    protected int m_OutputHeight;
    protected int m_OutputWidth;

    protected int m_WindowHeight;
    protected int m_WindowWidth;
    protected int m_StrideHeight;
    protected int m_StrideWidth;
    protected int m_PaddingHeight;
    protected int m_PaddingWidth;

    protected double[] m_Weights;

    #endregion

    #region .ctor

    protected DeepLayerBase(int outputDepth,
                            int windowSize,
                            int stride,
                            int padding=0,
                            IActivationFunction activation = null)
      : this(outputDepth,
             windowSize,
             windowSize,
             stride,
             stride,
             padding,
             padding,
             activation)
    {
    }

    protected DeepLayerBase(int outputDepth,
                            int windowHeight,
                            int windowWidth,
                            int strideHeight,
                            int strideWidth,
                            int paddingHeight=0,
                            int paddingWidth=0,
                            IActivationFunction activation = null)
    {
      if (outputDepth <= 0)
        throw new MLException("DeepLayerBase.ctor(outputDepth<=0)");
      if (windowHeight <= 0)
        throw new MLException("DeepLayerBase.ctor(windowHeight<=0)");
      if (windowWidth <= 0)
        throw new MLException("DeepLayerBase.ctor(windowWidth<=0)");
      if (strideHeight <= 0)
        throw new MLException("DeepLayerBase.ctor(strideHeight<=0)");
      if (strideWidth <= 0)
        throw new MLException("DeepLayerBase.ctor(strideWidth<=0)");
      if (paddingHeight < 0)
        throw new MLException("DeepLayerBase.ctor(paddingHeight<0)");
      if (paddingWidth < 0)
        throw new MLException("DeepLayerBase.ctor(paddingWidth<0)");

      m_WindowHeight  = windowHeight;
      m_WindowWidth   = windowWidth;
      m_StrideHeight  = strideHeight;
      m_StrideWidth   = strideWidth;
      m_PaddingHeight = paddingHeight;
      m_PaddingWidth  = paddingWidth;
      m_OutputDepth   = outputDepth;
      m_ActivationFunction = activation;
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_Weights.Length; } }

    public double[] Weights { get { return m_Weights; } }

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
    public int InputDepth
    {
      get { return m_InputDepth; }
      set
      {
        if (value <= 0)
          throw new MLException("Input depth must be positive");
        m_InputDepth=value;
      }
    }

    /// <summary>
    /// Height of input channel
    /// </summary>
    public int InputHeight
    {
      get { return m_InputHeight; }
      set
      {
        if (value <= 0)
          throw new MLException("Input height must be positive");
        m_InputHeight=value;
      }
    }

    /// <summary>
    /// Width of input channel
    /// </summary>
    public int InputWidth
    {
      get { return m_InputWidth; }
      set
      {
        if (value <= 0)
          throw new MLException("Input width must be positive");
        m_InputWidth=value;
      }
    }

    /// <summary>
    /// Count of output channels
    /// </summary>
    public int OutputDepth { get { return m_OutputDepth; } }

    /// <summary>
    /// Height of output channel
    /// </summary>
    public int OutputHeight { get { return m_OutputHeight; } }

    /// <summary>
    /// Width of output channel
    /// </summary>
    public int OutputWidth { get { return m_OutputWidth; } }

    /// <summary>
    /// Height of layer's window
    /// </summary>
    public int WindowHeight { get { return m_WindowHeight; } }

    /// <summary>
    /// Width of layer's window
    /// </summary>
    public int WindowWidth { get { return m_WindowWidth; } }

    /// <summary>
    /// An overlapping step in y direction during convolution calculation process.
    /// 1 leads to maximum overlapping between neighbour kernel windows
    /// </summary>
    public int StrideHeight { get { return m_StrideHeight; } }

    /// <summary>
    /// An overlapping step in x direction during convolution calculation process.
    /// 1 leads to maximum overlapping between neighbour kernel windows
    /// </summary>
    public int StrideWidth { get { return m_StrideWidth; } }

    /// <summary>
    /// Padding in y direction of the input channel
    /// </summary>
    public int PaddingHeight { get { return m_PaddingHeight; } }

    /// <summary>
    /// Padding in x direction of the input channel
    /// </summary>
    public int PaddingWidth { get { return m_PaddingWidth; } }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    #endregion

    public override double[][,] Calculate(double[][,] input)
    {
      var result = new double[m_OutputDepth][,];
      for (var q=0; q<m_OutputDepth; q++)
        result[q] = new double[m_OutputHeight, m_OutputWidth];

      Calculate(input, result);

      return result;
    }

    public void Calculate(double[][,] input, double[][,] result)
    {
      if (m_InputDepth != input.GetLength(0))
        throw new MLException("Incorrect input depth");

      DoCalculate(input, result);
    }

    /// <summary>
    /// Randomizes layer parameters (i.e. kernel weights, biases etc.)
    /// </summary>
    public virtual void RandomizeParameters(int seed)
    {
    }

    public virtual void _Build()
    {
      BuildShape();
      BuildParams();
    }

    protected virtual void BuildShape()
    {
      m_OutputHeight = (m_InputHeight - m_WindowHeight + 2*m_PaddingHeight)/m_StrideHeight + 1;
      m_OutputWidth  = (m_InputWidth  - m_WindowWidth  + 2*m_PaddingWidth)/m_StrideWidth   + 1;

      if (m_OutputHeight <= 0 || m_OutputWidth <= 0)
        throw new MLException("Output tensor is empty. Check input shape datas");
    }

    protected virtual void BuildParams()
    {
    }

    protected abstract void DoCalculate(double[][,] input, double[][,] result);

    /// <summary>
    /// Backpropagate "errors" to previous layer for future use
    /// </summary>
    /// <param name="prevLayer">Previous layer</param>
    /// <param name="errors">Current layer gradient "errors"</param>
    /// <param name="updates">Previous layer gradient "errors"</param>
    public void Backprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevErrors, double[][,] errors)
    {
      if (!m_IsTraining)
        throw new MLException("Backpropagation can not run in test mode");

      DoBackprop(prevLayer, prevValues, prevErrors, errors);
    }

    /// <summary>
    /// Calculate layer parameter updates
    /// </summary>
    /// <param name="prevLayer">Previous layer (or input layer)</param>
    /// <param name="errors">Current layer gradient "errors"</param>
    /// <param name="gradient">Current layer parameter gradient to copy into</param>
    /// <param name="isDelta">if true adds gradient values to existing ones, overwrites it otherwise</param>
    public void SetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
      if (!m_IsTraining)
        throw new MLException("Backpropagation can not run in test mode");

      DoSetLayerGradient(prevValues, errors, gradient, isDelta);
    }

    protected abstract void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevError, double[][,] errors);

    protected abstract void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta);

    protected override double DoGetParam(int idx)
    {
      return m_Weights[idx];
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      if (isDelta)
        m_Weights[idx] += value;
      else
        m_Weights[idx] = value;
    }

    protected override void DoUpdateParams(double[] updates, bool isDelta, int cursor)
    {
      var len = ParamCount;
      if (isDelta)
      {
        for (int i=0; i<len; i++)
          m_Weights[i] += updates[i+cursor];
      }
      else
        Array.Copy(updates, cursor, m_Weights, 0, len);
    }
  }
}
