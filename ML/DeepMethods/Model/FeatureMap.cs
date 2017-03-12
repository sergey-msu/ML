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
  /// Represents feature map - a convolution kernel neuron.
  /// Input - array of 2D martices:
  ///   input[0] - the first matrix
  ///   input[1] - the second matrix
  /// etc.
  ///   input[k][i, j] - element at i-th row, j-yh column of k-th matrix
  /// Output - 2D feature map received using convolution
  /// </summary>
  public class FeatureMap : NeuronNode<double[,]>
  {
    #region Fields

    private int m_WindowSize;
    private int m_WindowLength;
    private int m_InputSize;
    private int m_OutputSize;
    private int m_Stride;
    private int m_Padding;
    private int m_ParamCount;

    private double[,,] m_Kernel;

    #endregion

    #region .ctor

    public FeatureMap(int inputDim,
                      int inputSize,
                      int windowSize,
                      int stride=0,
                      int padding=0)
      : base(inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("FeatureMap.ctor(inputDim<=0)");
      if (inputSize <= 0 || windowSize <= 0)
        throw new MLException("FeatureMap.ctor(inputSize<=0|windowSize<=0)");
      if (windowSize > inputSize)
        throw new MLException("FeatureMap.ctor(windowSize>inputSize)");
      if (stride < 0)
        throw new MLException("FeatureMap.ctor(stride<0)");
      if (padding < 0)
        throw new MLException("FeatureMap.ctor(padding<0)");

      m_InputSize    = inputSize;
      m_WindowSize   = windowSize;
      m_WindowLength = windowSize*windowSize;
      m_Stride       = (stride == 0) ? m_WindowSize/2 : stride;
      m_Padding      = padding;
      m_OutputSize   = (inputSize+2*padding-windowSize)/m_Stride+1;
      m_ParamCount = inputDim*m_WindowLength+1;
      m_Kernel     = new double[m_WindowSize, m_WindowSize, inputDim];

      Value      = new double[m_OutputSize, m_OutputSize];
      NetValue   = new double[m_OutputSize, m_OutputSize];
      Derivative = new double[m_OutputSize, m_OutputSize];

    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_ParamCount; } }

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
    /// An overlapping step during convolution calculation process.
    /// 1 leads to maximum overlapping between neighour kernel windows
    /// 0 defaults to (int)m_WindowSize/2 (minimum overlapping that covers center)
    /// </summary>
    public int Padding { get { return m_Padding; } }

    public override double this[int idx]
    {
      get
      {
        var k = idx / m_WindowLength;
        var l = idx % m_WindowLength;
        var i = l / m_WindowSize;
        var j = l % m_WindowSize;
        return m_Kernel[i, j, k];
      }
      set
      {
        var k = idx / m_WindowLength;
        var l = idx % m_WindowLength;
        var i = l / m_WindowSize;
        var j = l % m_WindowSize;
        m_Kernel[i, j, k] = value;
      }
    }

    public double this[int i, int j, int k]
    {
      get { return m_Kernel[i, j, k]; }
      set { m_Kernel[i, j, k] = value; }
    }

    #endregion

    /// <summary>
    /// Randomizes feature map weights
    /// </summary>
    protected override void DoRandomizeParameters(RandomGenerator random)
    {
      var pcount = ParamCount;

      for (int i=0; i<m_WindowSize; i++)
      for (int j=0; j<m_WindowSize; j++)
      for (int k=0; k<InputDim; k++)
      {
        m_Kernel[i, j, k] = 2 * random.GenerateUniform(0, 1) / pcount;
      }

      Bias = 2 * random.GenerateUniform(0, 1) / pcount;
    }

    protected override void DoCalculate(double[][,] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorrect input dimensions");

      for (int i=0; i<m_OutputSize; i++)
      for (int j=0; j<m_OutputSize; j++)
      {
        var net = Bias;
        var xmin = j*m_Stride-m_Padding;
        var ymin = i*m_Stride-m_Padding;

        for (int k=0; k<InputDim; k++)
        for (int y=0; y<m_WindowSize; y++)
        for (int x=0; x<m_WindowSize; x++)
        {
          var xidx = x+xmin;
          var yidx = y+ymin;
          if (xidx>=0 && xidx<m_InputSize &&
              yidx>=0 && yidx<m_InputSize)
            net += m_Kernel[y, x, k]*input[k][yidx, xidx];
        }

        NetValue[i, j]   = net;
        Derivative[i, j] = ActivationFunction.Derivative(net);
        Value[i, j]      = ActivationFunction.Value(net);
      }
    }

    protected override double DoGetParam(int idx)
    {
      if (idx<m_ParamCount-1)
      {
        var k = idx / m_WindowLength;
        var l = idx % m_WindowLength;
        var i = l / m_WindowSize;
        var j = l % m_WindowSize;
        return m_Kernel[i, j, k];
      }

      if (idx==m_ParamCount-1) return Bias;

      throw new MLException("Index out of range");
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      if (idx<m_ParamCount-1)
      {
        var k = idx / m_WindowLength;
        var l = idx % m_WindowLength;
        var i = l / m_WindowSize;
        var j = l % m_WindowSize;
        if (isDelta)
          m_Kernel[i, j, k] += value;
        else
          m_Kernel[i, j, k] = value;
      }
      else if (idx==m_ParamCount-1)
      {
        if (isDelta)
          Bias += value;
        else
          Bias = value;
      }
      else
        throw new MLException("Index out of range");
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      var count = m_ParamCount-1;

      for (int idx=0; idx<count; idx++)
      {
        var k = idx / m_WindowLength;
        var l = idx % m_WindowLength;
        var i = l / m_WindowSize;
        var j = l % m_WindowSize;
        if (isDelta)
          m_Kernel[i, j, k] += pars[cursor++];
        else
          m_Kernel[i, j, k] = pars[cursor++];
      }

      if (isDelta)
        Bias += pars[cursor];
      else
        Bias = pars[cursor];
    }
  }
}
