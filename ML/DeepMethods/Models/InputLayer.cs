using System;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Fictive input layer
  /// </summary>
  public class InputLayer : DeepLayerBase
  {
    #region .ctor

    internal InputLayer()
      : base(outputDepth : 1, // will be overridden with input depth when building the layer
             windowSize : 1,
             stride : 1)
    {
    }

    #endregion

    public override int ParamCount { get { return 0; } }

    public override void _Build()
    {
      m_OutputDepth  = m_InputDepth;
      m_OutputHeight = m_InputHeight;
      m_OutputWidth  = m_InputWidth;
    }

    protected override void DoCalculate(double[][,] input, double[][,] result)
    {
      var len = result.Length;
      for (int i=0; i<len; i++)
        result[i] = input[i];
    }

    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevError, double[][,] errors)
    {
      throw new NotSupportedException();
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
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
}
