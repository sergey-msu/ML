using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.Core.Mathematics;

namespace ML.NeuralMethods.Model
{
  /// <summary>
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight).
  /// Neuron stores whole array of weights even if there are always-zero weights (connetion is not exist).
  /// </summary>
  public class Neuron : NeuronNode<double>
  {
    private double[] m_Weights;

    #region .ctor

    public Neuron(int inputDim) : base(inputDim)
    {
      m_Weights = new double[InputDim];
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_Weights.Length + 1; } }

    #endregion

    #region Public

    /// <summary>
    /// Calculates result value produced by neuron
    /// </summary>
    /// <param name="input">Input data array</param>
    protected override void DoCalculate(double[] input)
    {
      var net = Bias;

      for (int i=0; i<InputDim; i++)
        net += m_Weights[i] * input[i];

      Value = ActivationFunction.Value(net);
    }

    #endregion

    public override double this[int idx]
    {
      get { return m_Weights[idx]; }
      set { m_Weights[idx] = value; }
    }


    protected override void DoRandomizeParameters(RandomGenerator random)
    {
      for (int i=0; i<this.InputDim; i++)
        this[i] = 2 * random.GenerateUniform(0, 1) / InputDim;

      Bias = 2 * random.GenerateUniform(0, 1) / InputDim;
    }

    protected override double DoGetParam(int idx)
    {
      if (idx<InputDim)  return m_Weights[idx];
      if (idx==InputDim) return Bias;

      throw new MLException("Index out of range");
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      if (idx<InputDim)
      {
        if (isDelta)
          m_Weights[idx] += value;
        else
          m_Weights[idx] = value;
      }
      else if (idx==InputDim)
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
      var len = m_Weights.Length;
      if (isDelta)
      {
        for (int i=0; i<len; i++)
          m_Weights[i] += pars[cursor++];

        Bias += pars[cursor];
      }
      else
      {
        Array.Copy(pars, cursor, m_Weights, 0, len);
        cursor += len;
        Bias = pars[cursor];
      }
    }
  }
}
