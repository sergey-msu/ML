using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents pooling map (i.e. max pooling. average pooling etc.)
  /// </summary>
  public abstract class PoolingMap : ComputingNode<double[,], double[,]>
  {
    #region Fields

    protected int m_InputSize;
    protected int m_WindowSize;
    protected int m_OutputSize;
    protected int m_Stride;

    #endregion

    #region .ctor

    protected PoolingMap(int inputSize, int windowSize, int stride=0)
    {
      if (inputSize <= 0 || windowSize <= 0)
        throw new MLException("PoolingMap.ctor(inputSize<=0|windowSize<=0)");
      if (windowSize > inputSize)
        throw new MLException("PoolingMap.ctor(windowSize>inputSize)");
      if (stride < 0)
        throw new MLException("PoolingMap.ctor(stride<0)");

      m_InputSize  = inputSize;
      m_WindowSize = windowSize;
      m_Stride     = (stride == 0) ? m_WindowSize : stride;
      m_OutputSize = (m_InputSize-m_WindowSize)/m_Stride+1;
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return 0; } }

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

    #endregion

    protected override double DoGetParam(int idx)
    {
      throw new NotSupportedException();
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      throw new NotSupportedException();
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      throw new NotSupportedException();
    }
  }

  /// <summary>
  /// Represents max pooling operation
  /// </summary>
  public class MaxPoolingMap : PoolingMap
  {
    #region .ctor

    public MaxPoolingMap(int inputSize, int windowSize, int stride=0)
      : base(inputSize, windowSize, stride)
    {
    }

    #endregion


    public override double[,] Calculate(double[,] input)
    {
      var result = new double[m_OutputSize, m_OutputSize];

      for (int i=0; i<m_OutputSize; i++)
      for (int j=0; j<m_OutputSize; j++)
      {
        var value = double.MinValue;
        var xmin = j*m_Stride;
        var xmax = xmin+m_WindowSize;
        var ymin = i*m_Stride;
        var ymax = ymin+m_WindowSize;

        for (int y=ymin; y<ymax; y++)
        for (int x=xmin; x<xmax; x++)
        {
          var z = input[y, x];
          if (z > value) value = z;
        }

        result[i, j] = value;
      }

      return result;
    }
  }

  /// <summary>
  /// Represents max pooling operation
  /// </summary>
  public class AvgPoolingMap : PoolingMap
  {
    #region .ctor

    public AvgPoolingMap(int inputSize, int windowSize, int stride=0)
      : base(inputSize, windowSize, stride)
    {
    }

    #endregion


    public override double[,] Calculate(double[,] input)
    {
      var result = new double[m_OutputSize, m_OutputSize];
      var l = m_OutputSize*m_OutputSize;

      for (int i=0; i<m_OutputSize; i++)
      for (int j=0; j<m_OutputSize; j++)
      {
        var value = 0.0D;
        var xmin = j*m_Stride;
        var xmax = xmin+m_WindowSize;
        var ymin = i*m_Stride;
        var ymax = ymin+m_WindowSize;

        for (int y=ymin; y<ymax; y++)
        for (int x=xmin; x<xmax; x++)
        {
          value += input[y, x];
        }

        result[i, j] = value/l;
      }

      return result;
    }
  }
}
