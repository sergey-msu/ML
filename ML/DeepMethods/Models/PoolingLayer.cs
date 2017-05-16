using System;
using System.Threading;
using ML.Contracts;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents pooling layer (i.e. max pooling, average pooling etc.)
  /// </summary>
  public abstract class PoolingLayer : DeepLayerBase
  {
    protected int m_KernelArea;

    #region .ctor

    protected PoolingLayer(int windowSize,
                           int stride,
                           int padding=0,
                           IActivationFunction activation = null)
      : base(1, // will be overridden with input depth when building the layer
             windowSize,
             stride,
             padding,
             activation)
    {
    }

    protected PoolingLayer(int windowHeight,
                           int windowWidth,
                           int strideHeight,
                           int strideWidth,
                           int paddingHeight=0,
                           int paddingWidth=0,
                           IActivationFunction activation = null)
      : base(1, // will be overridden with input depth when building the layer
             windowHeight,
             windowWidth,
             strideHeight,
             strideWidth,
             paddingHeight,
             paddingWidth,
             activation)
    {
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return 0; } }

    #endregion

    protected override void BuildShape()
    {
      m_OutputDepth = m_InputDepth;
      m_KernelArea = m_WindowHeight*m_WindowWidth;

      base.BuildShape();
    }

    protected override double DoGetParam(int idx)
    {
      return 0;
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
    }
  }

  /// <summary>
  /// Represents max pooling layer
  /// </summary>
  public class MaxPoolingLayer : PoolingLayer
  {
    [NonSerialized]
    private object m_Sync = new object();

    [NonSerialized]
    private ThreadLocal<int[][,,]> m_MaxIndexPositions;

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

    public MaxPoolingLayer(int windowHeight,
                           int windowWidth,
                           int strideHeight,
                           int strideWidth,
                           int paddingHeight=0,
                           int paddingWidth=0,
                           IActivationFunction activation = null)
      : base(windowHeight,
             windowWidth,
             strideHeight,
             strideWidth,
             paddingHeight,
             paddingWidth,
             activation)
    {
    }

    #endregion

    private ThreadLocal<int[][,,]> MaxIndexPositions
    {
      get
      {
        if (m_MaxIndexPositions==null)
        {
          lock (m_Sync)
          {
            if (m_MaxIndexPositions==null)
            {
              m_MaxIndexPositions = new ThreadLocal<int[][,,]>(() =>
              {
                var maxPos = new int[m_OutputDepth][,,];
                for (int i=0; i<m_OutputDepth; i++)
                  maxPos[i] = new int[m_OutputHeight, m_OutputWidth, 2];
                return maxPos;
              });
            }
          }
        }

        return m_MaxIndexPositions;
      }
    }


    protected override void DoCalculate(double[][,] input, double[][,] result)
    {
      var maxPos = MaxIndexPositions;

      for (int q=0; q<m_OutputDepth; q++)
      {
        for (int i=0; i<m_OutputHeight; i++)
        for (int j=0; j<m_OutputWidth; j++)
        {
          var net = double.MinValue;
          var xmaxIdx = -1;
          var ymaxIdx = -1;
          var xmin = j*m_StrideWidth-m_PaddingWidth;
          var ymin = i*m_StrideHeight-m_PaddingHeight;

          // window
          for (int y=0; y<m_WindowHeight; y++)
          {
            var yidx = ymin+y;
            if (yidx<0) continue;
            if (yidx>=m_InputHeight) break;

            for (int x=0; x<m_WindowWidth; x++)
            {
              var xidx = xmin+x;
              if (xidx<0) continue;
              if (xidx>=m_InputWidth) break;

              var z = input[q][yidx, xidx];
              if (z > net)
              {
                net = z;
                xmaxIdx = xidx;
                ymaxIdx = yidx;
              }
            }
          }

          result[q][i, j] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;

          if (m_IsTraining)
          {
            maxPos.Value[q][i, j, 0] = xmaxIdx;
            maxPos.Value[q][i, j, 1] = ymaxIdx;
          }
        }
      }
    }

    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevErrors, double[][,] errors)
    {
      for (int i=0; i<prevErrors.Length; i++)
        Array.Clear(prevErrors[i], 0, prevErrors[i].Length);

      var maxPos = MaxIndexPositions;

      // backpropagate "errors" to previous layer for future use
      for (int q=0; q<m_OutputDepth;  q++)
      for (int i=0; i<m_OutputHeight; i++)
      for (int j=0; j<m_OutputWidth;  j++)
      {
        var xmaxIdx = maxPos.Value[q][i, j, 0];
        var ymaxIdx = maxPos.Value[q][i, j, 1];
        var value = prevValues[q][ymaxIdx, xmaxIdx];
        var deriv = (prevLayer.ActivationFunction != null) ? prevLayer.ActivationFunction.DerivativeFromValue(value) : 1;
        prevErrors[q][ymaxIdx, xmaxIdx] += errors[q][i, j] * deriv;
      }
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
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

    public AvgPoolingLayer(int windowHeight,
                           int windowWidth,
                           int strideHeight,
                           int strideWidth,
                           int paddingHeight=0,
                           int paddingWidth=0,
                           IActivationFunction activation = null)
      : base(windowHeight,
             windowWidth,
             strideHeight,
             strideWidth,
             paddingHeight,
             paddingWidth,
             activation)
    {
    }

    #endregion

    protected override void DoCalculate(double[][,] input, double[][,] result)
    {
      // output fm-s
      for (int q=0; q<m_OutputDepth; q++)
      {
        // fm neurons
        for (int i=0; i<m_OutputHeight; i++)
        for (int j=0; j<m_OutputWidth; j++)
        {
          var net = 0.0D;
          var xmin = j*m_StrideWidth-m_PaddingWidth;
          var ymin = i*m_StrideHeight-m_PaddingHeight;

          // window
          for (int y=0; y<m_WindowHeight; y++)
          for (int x=0; x<m_WindowWidth; x++)
          {
            var xidx = xmin+x;
            var yidx = ymin+y;
            if (xidx>=0 && xidx<m_InputWidth && yidx>=0 && yidx<m_InputHeight)
            {
              net += input[q][yidx, xidx];
            }
          }

          net /= m_KernelArea;

          result[q][i, j] = (m_ActivationFunction != null) ? m_ActivationFunction.Value(net) : net;
        }
      }
    }

    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevErrors, double[][,] errors)
    {
      throw new NotImplementedException(); //TODO
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
    }
  }
}
