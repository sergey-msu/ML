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

    private IFunction m_ActivationFunction;
    private int    m_InputDim;
    private double m_Bias;
    private TPar   m_NetValue;
    private TPar   m_Derivative;
    private TPar   m_Value;
    private TPar   m_Error;

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
    public TPar NetValue
    {
      get { return m_NetValue; }
      set { m_NetValue = value; }
    }

    /// <summary>
    /// Cached derivative of pure calculated value
    /// </summary>
    public TPar Derivative
    {
      get { return m_Derivative; }
      set { m_Derivative = value; }
    }

    /// <summary>
    /// Calculated value (after applying activation function)
    /// </summary>
    public TPar Value
    {
      get { return m_Value; }
      set { m_Value = value; }
    }

    /// <summary>
    /// Calculated error
    /// </summary>
    public TPar Error
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
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for connection weights.
    /// </summary>
    public abstract double this[int idx] { get; set; }

    #endregion

    #region Public

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
