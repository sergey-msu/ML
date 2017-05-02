using System;
using ML.Core;
using ML.Contracts;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Applies activation function to input
  /// </summary>
  public class ActivationLayer : DeepLayerBase
  {
    #region .ctor

    public ActivationLayer(IActivationFunction activation)
      : base(outputDepth: 1, // will be overridden with input depth when building the layer
             windowSize: 1,
             stride: 1,
             padding: 0,
             activation: activation)
    {
      if (m_ActivationFunction==null)
        throw new MLException("Activation function is null");
    }

    #endregion

    public override int ParamCount { get { return 0; } }


    protected override void DoCalculate(double[][,] input, double[][,] result)
    {
      for (int p=0; p<m_InputDepth;  p++)
      for (int i=0; i<m_InputHeight; i++)
      for (int j=0; j<m_InputWidth;  j++)
      {
        result[p][i, j] = m_ActivationFunction.Value(input[p][i, j]);
      }
    }

    public override void _Build()
    {
      m_OutputDepth = m_InputDepth;

      base._Build();
    }

    protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevError, double[][,] errors)
    {
      for (int p=0; p<m_OutputDepth;  p++)
      for (int i=0; i<m_OutputHeight; i++)
      for (int j=0; j<m_OutputWidth;  j++)
      {
        var value = prevValues[p][i,j];
        var deriv = (prevLayer.ActivationFunction != null) ? prevLayer.ActivationFunction.DerivativeFromValue(value) : 1;
        prevError[p][i, j] = errors[p][i, j] * deriv;
      }
    }

    protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
    {
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
}
