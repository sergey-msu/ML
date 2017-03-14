using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents pooling layer (i.e. max pooling, average pooling etc.)
  /// </summary>
  public abstract class PoolingLayer : DeepLayerBase
  {
    #region Fields

    protected readonly int m_WindowSize;
    protected readonly int m_Stride;
    protected readonly int m_Padding;

    #endregion

    #region .ctor

    protected PoolingLayer(int inputDepth,
                           int inputSize,
                           int windowSize,
                           int stride=0,
                           int padding=0)
      : base(inputDepth,
             inputSize,
             inputDepth,
             (inputSize - windowSize + 2*padding)/stride + 1)
    {
      if (windowSize <= 0)
        throw new MLException("PoolingLayer.ctor(windowSize<=0)");
      if (windowSize > inputSize)
        throw new MLException("PoolingLayer.ctor(windowSize>inputSize)");
      if (stride < 0)
        throw new MLException("PoolingLayer.ctor(stride<0)");
      if (padding < 0)
        throw new MLException("PoolingLayer.ctor(padding<0)");

      m_WindowSize = windowSize;
      m_Padding    = padding;
      m_Stride     = (stride == 0) ? windowSize : stride;
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return 0; } }

    /// <summary>
    /// Size of square pooling window
    /// </summary>
    public int WindowSize { get { return m_WindowSize; } }

    /// <summary>
    /// An overlapping step during pooling process.
    /// 1 leads to maximum overlapping between neighour windows
    /// </summary>
    public int Stride { get { return m_Stride; } }

    /// <summary>
    /// Padding of the input channel
    /// </summary>
    public int Padding { get { return m_Padding; } }

    #endregion

    public override void RandomizeParameters(int seed)
    {
    }

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
  public class MaxPoolingLayer : PoolingLayer
  {
    #region .ctor

    public MaxPoolingLayer(int inputDepth,
                           int inputSize,
                           int windowSize,
                           int stride=0,
                           int padding=0)
      : base(inputDepth,
             inputSize,
             windowSize,
             stride,
             padding)
    {
    }

    #endregion

    public override double[,,] Calculate(double[,,] input)
    {
      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int i=0; i<m_OutputSize; i++)
        for (int j=0; j<m_OutputSize; j++)
        {
          var value = double.MinValue;
          var xmin = j*m_Stride-m_Padding;
          var ymin = i*m_Stride-m_Padding;

          // window
          for (int y=0; y<m_WindowSize; y++)
          for (int x=0; x<m_WindowSize; x++)
          {
            var xidx = xmin+x;
            var yidx = ymin+y;
            if (xidx>=0 && xidx<m_InputSize && yidx>=0 && yidx<m_InputSize)
            {
              var z = input[q, yidx, xidx];
              if (z > value) value = z;
            }
          }

          m_Value[q, i, j] = value;
        }
      }

      return m_Value;
    }
  }

  /// <summary>
  /// Represents max pooling operation
  /// </summary>
  public class AvgPoolingLayer : PoolingLayer
  {
    #region .ctor

    public AvgPoolingLayer(int inputDepth,
                           int inputSize,
                           int windowSize,
                           int stride=0,
                           int padding=0)
      : base(inputDepth,
             inputSize,
             windowSize,
             stride,
             padding)
    {
    }

    #endregion

    public override double[,,] Calculate(double[,,] input)
    {
      var l = m_WindowSize*m_WindowSize;

      // output fm-s
      for (int q=0; q<m_OutputDepth; q++)
      {
        // fm neurons
        for (int i=0; i<m_OutputSize; i++)
        for (int j=0; j<m_OutputSize; j++)
        {
          var value = 0.0D;
          var xmin = j*m_Stride-m_Padding;
          var ymin = i*m_Stride-m_Padding;

          // window
          for (int y=0; y<m_WindowSize; y++)
          for (int x=0; x<m_WindowSize; x++)
          {
            var xidx = xmin+x;
            var yidx = ymin+y;
            if (xidx>=0 && xidx<m_InputSize && yidx>=0 && yidx<m_InputSize)
            {
              value += input[q, yidx, xidx];
            }
          }

          m_Value[q, i, j] = value/l;
        }
      }

      return m_Value;
    }
  }
}
