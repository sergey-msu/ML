using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ML.Contracts;
using ML.Core.Mathematics;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents abstract artificial neuron
  /// </summary>
  public abstract class NeuronNode<TPar> : ComputingNode<TPar[], TPar>
  {
    #region Fields

    protected IFunction m_ActivationFunction;
    protected int  m_InputDim;
    protected double m_Bias;
    protected TPar m_NetValue;
    protected TPar m_Derivative;
    protected TPar m_Value;

    #endregion

    #region .ctor

    protected NeuronNode(int inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("NeuronNode.ctor(inputDim<=0)");

      m_InputDim = inputDim;
    }

    #endregion

    #region Properties

    /// <summary>
    /// Bias weight value
    /// </summary>
    public double Bias
    {
      get { return m_Bias; }
      set { m_Bias = value; }
    }

    /// <summary>
    /// Calculated pure value (before applying activation function)
    /// </summary>
    public TPar NetValue { get { return m_NetValue; } }

    /// <summary>
    /// Cached derivative of pure calculated value
    /// </summary>
    public TPar Derivative { get { return m_Derivative; } }

    /// <summary>
    /// Calculated value (after applying activation function)
    /// </summary>
    public TPar Value { get { return m_Value; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim
    {
      get { return m_InputDim; }
      internal set { m_InputDim = value; }
    }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for connection weights.
    /// Warning! idx here is an index of neuron in the previous layer NOT SERIAL (indexed) parameter index.
    /// Neuron may be connected with 0,1,4,7,8-th neurons in the previous layer,
    /// while through parameter index will be serial: 0,1,2,3,4
    /// (therefore one can not use DoGetParam/DoSetParam methods as they operate with serial indexed parameter indices).
    /// </summary>
    public abstract double this[int idx] { get; set; }

    #endregion

    #region Public

    /// <summary>
    /// Insert new synapse weight at specified index
    /// </summary>
    //public abstract void AddSynapse(double weight = 0.0D);

    /// <summary>
    /// Randomizes neuron weights
    /// </summary>
    public void RandomizeParameters(int seed=0)
    {
      var random = RandomGenerator.Get(seed);
      DoRandomizeParameters(random);
    }

    /// <summary>
    /// Calculates result value produced by neuron
    /// </summary>
    /// <param name="input">Input data array</param>
    public override TPar Calculate(TPar[] input)
    {
      if (m_InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      DoCalculate(input);

      return m_Value;
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      base.DoBuild();
    }

    #endregion

    #region Protected

    protected abstract void DoRandomizeParameters(RandomGenerator random);

    protected abstract void DoCalculate(TPar[] input);

    #endregion
  }

}
