using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.Core.Mathematics;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents convolution kernel neuron.
  /// Input - array of 2D martices:
  ///   input[0, *, *] - the first matrix
  ///   input[1, *, *] - the second matrix
  /// etc.
  ///   input[k, i, j] - element at i-th row, j-yh column of k-th matrix
  /// Output - 2D feature map received using convolution
  /// </summary>
  public class KernelNeuron : ComputingNode<double[,,], double[,]>
  {
    #region Fields

    private IFunction m_ActivationFunction;
    private int m_InputDim;
    private int m_WindowHeight;
    private int m_WindowWidth;
    private int m_InputHeight;
    private int m_InputWidth;
    private int m_OutputHeight;
    private int m_OutputWidth;
    private int m_WidthStride;
    private int m_HeightStride;
    private int m_ParamCount;
    private double[,] m_Kernel;

    private double[,] m_NetValues;
    private double[,] m_Derivatives;
    private double[,] m_Values;

    private double m_Bias;

    #endregion

    #region .ctor

    public KernelNeuron(int inputDim,
                        int inputHeight,
                        int inputWidth,
                        int windowHeight,
                        int windowWidth,
                        int heightStride=0,
                        int widthStride=0)
    {
      if (inputDim <= 0)
        throw new MLException("KernelNeuron.ctor(inputDim<=0)");
      if (inputWidth <= 0 || inputHeight <= 0 || windowWidth <= 0 || windowHeight <= 0)
        throw new MLException("KernelNeuron.ctor(inputWidth<=0|inputHeight<=0|windowWidth<=0|windowHeight<=0)");
      if (inputWidth <= 0 || inputHeight <= 0 || windowWidth <= 0 || windowHeight <= 0)
        throw new MLException("KernelNeuron.ctor(inputWidth<=0|inputHeight<=0|windowWidth<= 0|windowHeight<=0)");
      if (windowWidth > inputWidth)
        throw new MLException("KernelNeuron.ctor(windowWidth>inputWidth)");
      if (windowHeight > inputHeight)
        throw new MLException("KernelNeuron.ctor(windowHeight>inputHeight)");
      if (windowWidth > inputWidth)
        throw new MLException("KernelNeuron.ctor(windowWidth>inputWidth)");
      if (windowHeight > inputHeight)
        throw new MLException("KernelNeuron.ctor(windowHeight>inputHeight)");
      if (heightStride < 0 || widthStride < 0)
        throw new MLException("KernelNeuron.ctor(heightStride<0|widthStride<0)");

      m_InputDim = inputDim;
      m_InputHeight  = inputHeight;
      m_InputWidth   = inputWidth;
      m_WindowHeight = windowHeight;
      m_WindowWidth  = windowWidth;
      m_WidthStride  = widthStride;
      m_HeightStride = heightStride;

      if (m_WidthStride==0)  m_WidthStride = m_WindowWidth/2;
      if (m_HeightStride==0) m_HeightStride= m_WindowHeight/2;

      m_Kernel = new double[m_WindowHeight, m_WindowWidth];

      m_OutputWidth  = (m_InputWidth-m_WindowWidth)/m_WidthStride+1;
      m_OutputHeight = (m_InputHeight-m_WindowHeight)/m_HeightStride+1;

      m_Values       = new double[m_OutputHeight, m_OutputWidth];
      m_NetValues    = new double[m_OutputHeight, m_OutputWidth];
      m_Derivatives  = new double[m_OutputHeight, m_OutputWidth];

      m_ParamCount = m_WindowHeight*m_WindowWidth+1;
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_ParamCount; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get { return m_InputDim; } }

    public int WindowHeight { get { return m_WindowHeight; } }

    public int WindowWidth { get { return m_WindowWidth; } }

    public int InputHeight { get { return m_InputHeight; } }

    public int InputWidth { get { return m_InputWidth; } }

    public int OutputHeight { get { return m_OutputHeight; }}

    public int OutputWidth { get { return m_OutputWidth; }}

    /// <summary>
    /// Calculated pure value (before applying activation function)
    /// </summary>
    public double[,] NetValues { get { return m_NetValues; } }

    /// <summary>
    /// Cached derivative of pure calculated value
    /// </summary>
    public double[,] Derivatives { get { return m_Derivatives; } }

    /// <summary>
    /// Calculated value (after applying activation function)
    /// </summary>
    public double[,] Values { get { return m_Values; } }

    /// <summary>
    /// Activation function. If null, identity map x -> x will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// A x-overlapping step during convolution calculation process.
    /// 1 leads to maximum overlapping between neighour kernel windows
    /// 0 defaults to (int)m_WindowWidth/2 (minimum overlapping that covers center)
    /// </summary>
    public int WidthStride { get { return m_WidthStride; } }

    /// <summary>
    /// A y-overlapping step during convolution calculation process.
    /// 1 leads to maximum overlapping between neighour kernel windows
    /// 0 defaults to (int)m_WindowHeight/2 (minimum overlapping that covers center)
    /// </summary>
    public int HeightStride { get { return m_HeightStride; } }

    /// <summary>
    /// Bias weight value
    /// </summary>
    public double Bias
    {
      get { return m_Bias; }
      set { m_Bias = value; }
    }

    public double this[int idx]
    {
      get { return m_Kernel[idx / m_WindowWidth, idx % m_WindowWidth]; }
      set { m_Kernel[idx /m_WindowWidth, idx % m_WindowWidth] = value; }
    }

    public double this[int i, int j]
    {
      get { return m_Kernel[i, j]; }
      set { m_Kernel[i, j] = value; }
    }

    #endregion

    /// <summary>
    /// Randomizes neuron weights
    /// </summary>
    public void Randomize(int seed=0)
    {
      var random = RandomGenerator.Get(seed);
      var pcount = ParamCount;

      for (int i=0; i<m_WindowWidth;  i++)
      for (int j=0; j<m_WindowHeight; j++)
      {
        m_Kernel[i, j] = 2 * random.GenerateUniform(0, 1) / pcount;
      }

      m_Bias = 2 * random.GenerateUniform(0, 1) / pcount;
    }

    public override void DoBuild()
    {
      base.DoBuild();

      if (m_ActivationFunction == null)
        m_ActivationFunction = Registry.ActivationFunctions.ReLU;
    }

    public override double[,] Calculate(double[,,] input)
    {
      if (m_InputDim != input.GetLength(0) ||
          m_InputHeight != input.GetLength(1) ||
          m_InputWidth != input.GetLength(2))
        throw new MLException("Incorrect input dimensions");

      for (int i=0; i<m_OutputHeight; i++)
      for (int j=0; j<m_OutputWidth;  j++)
      {
        var value = m_Bias;
        var xmin = j*m_WidthStride;
        var xmax = xmin + m_WindowWidth;
        var ymin = i*m_HeightStride;
        var ymax = ymin + m_WindowHeight;

        for (int k=0; k<m_InputDim; k++)
        for (int y=ymin; y<ymax; y++)
        for (int x=xmin; x<xmax; x++)
        {
          value += m_Kernel[y-ymin, x-xmin]*input[k, y, x];
        }

        m_NetValues[i, j]   = value;
        m_Derivatives[i, j] = m_ActivationFunction.Derivative(value);
        m_Values[i, j]      = m_ActivationFunction.Value(value);
      }

      return m_Values;
    }

    protected override double DoGetParam(int idx)
    {
      if (idx<m_ParamCount-1)
      {
        var i = idx / m_WindowWidth;
        var j = idx % m_WindowWidth;
        return m_Kernel[i, j];
      }

      if (idx==m_ParamCount-1) return m_Bias;

      throw new MLException("Index out of range");
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      if (idx<m_ParamCount-1)
      {
        var i = idx / m_WindowWidth;
        var j = idx % m_WindowWidth;
        if (isDelta)
          m_Kernel[i, j] += value;
        else
          m_Kernel[i, j] = value;
      }
      else if (idx==m_ParamCount-1)
      {
        if (isDelta)
          m_Bias += value;
        else
          m_Bias = value;
      }
      else
        throw new MLException("Index out of range");
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      var count = m_ParamCount-1;

      for (int idx=0; idx<count; idx++)
      {
        var i = idx / m_WindowWidth;
        var j = idx % m_WindowWidth;
        if (isDelta)
          m_Kernel[i, j] += pars[cursor++];
        else
          m_Kernel[i, j] = pars[cursor++];
      }

      if (isDelta)
        m_Bias += pars[cursor];
      else
        m_Bias = pars[cursor];
    }
  }
}
