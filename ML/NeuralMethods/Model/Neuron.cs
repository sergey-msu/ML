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
      m_Weights = new double[m_InputDim];
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
      var net = m_Bias;

      for (int i=0; i<m_InputDim; i++)
        net += m_Weights[i] * input[i];

      m_NetValue   = net;
      m_Value      = m_ActivationFunction.Value(net);
      m_Derivative = m_ActivationFunction.Derivative(net);
    }

    #endregion

    public override double this[int idx]
    {
      get { return m_Weights[idx]; }
      set { m_Weights[idx] = value; }
    }

    /// <summary>
    /// Insert new synapse weight at specified index
    /// </summary>
    //public override void AddSynapse(double weight = 0.0D)
    //{
    //  var weights = new double[m_InputDim];
    //
    //  Array.Copy(m_Weights, weights, m_InputDim-1);
    //  weights[m_InputDim-1] = weight;
    //
    //  m_Weights = weights;
    //}

    protected override void DoRandomizeParameters(RandomGenerator random)
    {
      for (int i=0; i<this.InputDim; i++)
        this[i] = 2 * random.GenerateUniform(0, 1) / m_InputDim;

      m_Bias = 2 * random.GenerateUniform(0, 1) / m_InputDim;
    }

    protected override double DoGetParam(int idx)
    {
      if (idx<m_InputDim)  return m_Weights[idx];
      if (idx==m_InputDim) return m_Bias;

      throw new MLException("Index out of range");
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      if (idx<m_InputDim)
      {
        if (isDelta)
          m_Weights[idx] += value;
        else
          m_Weights[idx] = value;
      }
      else if (idx==m_InputDim)
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
      var len = m_Weights.Length;
      if (isDelta)
      {
        for (int i=0; i<len; i++)
          m_Weights[i] += pars[cursor++];

        m_Bias += pars[cursor];
      }
      else
      {
        Array.Copy(pars, cursor, m_Weights, 0, len);
        cursor += len;
        m_Bias = pars[cursor];
      }
    }
  }
}
