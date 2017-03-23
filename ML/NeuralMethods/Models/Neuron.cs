using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.Core.Mathematics;

namespace ML.NeuralMethods.Models
{
  /// <summary>
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight).
  /// Neuron stores whole array of weights even if there are always-zero weights (connetion is not exist).
  /// </summary>
  public class Neuron : ComputingNode<double[], double>
  {
    #region Fields

    private IActivationFunction m_ActivationFunction;
    private int      m_InputDim;
    private double   m_Bias;
    private double   m_Value;
    private double   m_Error;
    private double[] m_Weights;

    #endregion

    #region .ctor

    public Neuron(int inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("NeuronNode.ctor(inputDim<=0)");

      m_InputDim = inputDim;
      m_Weights = new double[inputDim];
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return m_Weights.Length + 1; } }

    /// <summary>
    /// Bias weight value
    /// </summary>
    public double Bias
    {
      get { return m_Bias; }
      set { m_Bias = value; }
    }

    /// <summary>
    /// Calculated value (after applying activation function)
    /// </summary>
    public double Value
    {
      get { return m_Value; }
      set { m_Value = value; }
    }

    /// <summary>
    /// Calculates derivative
    /// </summary>
    public double Derivative
    {
      get { return ActivationFunction.DerivativeFromValue(m_Value); }
    }

    /// <summary>
    /// Calculated error
    /// </summary>
    public double Error
    {
      get { return m_Error; }
      set { m_Error = value; }
    }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim
    {
      get { return m_InputDim; }
    }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for connection weights.
    /// </summary>

    public double this[int idx]
    {
      get { return m_Weights[idx]; }
      set { m_Weights[idx] = value; }
    }

    #endregion

    #region Public

    /// <summary>
    /// Randomizes neuron weights
    /// </summary>
    public void RandomizeParameters(int seed=0)
    {
      var random = RandomGenerator.Get(seed);

      for (int i=0; i<m_InputDim; i++)
        m_Weights[i] = 2 * random.GenerateUniform(0, 1) / m_InputDim;

      m_Bias = 2 * random.GenerateUniform(0, 1) / m_InputDim;
    }

    /// <summary>
    /// Calculates result value produced by neuron
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double Calculate(double[] input)
    {
      if (m_InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      var net = m_Bias;

      for (int i=0; i<InputDim; i++)
        net += m_Weights[i] * input[i];

      m_Value = ActivationFunction.Value(net);

      return m_Value;
    }

    public override void DoBuild()
    {
      if (m_InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      base.DoBuild();
    }

    #endregion

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
