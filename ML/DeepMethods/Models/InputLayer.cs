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

    public override double[][,] Calculate(double[][,] input)
    {
      m_Value = input;
      return input;
    }

    protected override double[][,] DoCalculate(double[][,] input)
    {
      throw new NotSupportedException();
    }

    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] error, double[][,] prevError)
    {
      throw new NotSupportedException();
    }

    protected override void DoSetLayerGradient(DeepLayerBase prevLayer, double[][,] errors, double[] updates)
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
