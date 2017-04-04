using System;
using ML.Contracts;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents pooling layer (i.e. max pooling, average pooling etc.)
  /// </summary>
  public abstract class PoolingLayer : DeepLayerBase
  {
    #region .ctor

    protected PoolingLayer(int windowSize,
                           int stride,
                           int padding=0,
                           IActivationFunction activation = null)
      : base(1, // to be overridden with input depth on build
             windowSize,
             stride,
             padding,
             activation)
    {
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return 0; } }

    #endregion

    public override void DoBuild()
    {
      m_OutputDepth = m_InputDepth;

      base.DoBuild();
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
    private int[,,,] m_MaxIndexPositions;

    #region .ctor

    public MaxPoolingLayer(int windowSize,
                           int stride,
                           int padding=0,
                           IActivationFunction activation = null)
      : base(windowSize,
             stride,
             padding,
             activation)
    {
    }

    #endregion

    public int[,,,] MaxIndexPositions { get { return m_MaxIndexPositions; } }

    protected override double[,,] DoCalculate(double[,,] input)
    {
      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int i=0; i<m_OutputSize; i++)
        for (int j=0; j<m_OutputSize; j++)
        {
          var net = double.MinValue;
          var xmaxIdx = -1;
          var ymaxIdx = -1;
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
              if (z > net)
              {
                net = z;
                xmaxIdx = xidx;
                ymaxIdx = yidx;
              }
            }
          }

          m_Value[q, i, j] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;
          m_MaxIndexPositions[q, i, j, 0] = xmaxIdx;
          m_MaxIndexPositions[q, i, j, 1] = ymaxIdx;
        }
      }

      return m_Value;
    }

    public override void DoBuild()
    {
      base.DoBuild();

      m_MaxIndexPositions = new int[m_OutputDepth, m_OutputSize, m_OutputSize, 2];
    }
  }

  /// <summary>
  /// Represents max pooling operation
  /// </summary>
  public class AvgPoolingLayer : PoolingLayer
  {
    #region .ctor

    public AvgPoolingLayer(int windowSize,
                           int stride,
                           int padding=0,
                           IActivationFunction activation = null)
      : base(windowSize,
             stride,
             padding,
             activation)
    {
    }

    #endregion

    protected override double[,,] DoCalculate(double[,,] input)
    {
      var l = m_WindowSize*m_WindowSize;

      // output fm-s
      for (int q=0; q<m_OutputDepth; q++)
      {
        // fm neurons
        for (int i=0; i<m_OutputSize; i++)
        for (int j=0; j<m_OutputSize; j++)
        {
          var net = 0.0D;
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
              net += input[q, yidx, xidx];
            }
          }

          net /= l;

          m_Value[q, i, j] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;
        }
      }

      return m_Value;
    }
  }
}
