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
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight)
  /// </summary>
  public abstract class Neuron : ComputingNode<double[], double>
  {
    #region Fields

    private readonly NeuralLayer m_Layer;
    private IFunction m_ActivationFunction;
    protected int     m_InputDim;
    protected double  m_Bias;

    private double m_NetValue;
    private double m_Derivative;
    private double m_Value;

    #endregion

    #region .ctor

    public Neuron(int inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("Neuron.ctor(inputDim<=0)");

      m_InputDim = inputDim;
    }

    public Neuron(NeuralLayer layer)
      : this(layer.InputDim)
    {
      if (layer==null)
        throw new MLException("Neuron.ctor(layer=null)");

      m_Layer = layer;
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
    public double NetValue { get { return m_NetValue; } }

    /// <summary>
    /// Cached derivative of pure calculated value
    /// </summary>
    public double Derivative { get { return m_Derivative; } }

    /// <summary>
    /// Calculated value (after applying activation function)
    /// </summary>
    public double Value { get { return m_Value; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim
    {
      get { return m_InputDim; }
      internal set { m_InputDim = value; }
    }

    /// <summary>
    /// Layer which is neuron belongs to
    /// </summary>
    public NeuralLayer Layer { get { return m_Layer; } }

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
    public abstract void AddSynapse(double weight = 0.0D);

    /// <summary>
    /// Randomizes neuron weights
    /// </summary>
    public void Randomize(int seed=0)
    {
      var random = RandomGenerator.Get(seed);
      DoRandomize(random);
    }

    /// <summary>
    /// Calculates result value produced by neuron
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double Calculate(double[] input)
    {
      if (m_InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      m_NetValue   = DoCalculate(input);
      m_Derivative = m_ActivationFunction.Derivative(m_NetValue);
      m_Value      = m_ActivationFunction.Value(m_NetValue);

      return m_Value;
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      if (ActivationFunction==null)
        ActivationFunction = (Layer != null ? Layer.ActivationFunction : null) ?? Registry.ActivationFunctions.Identity;

      base.DoBuild();
    }

    #endregion

    #region Protected

    protected abstract void DoRandomize(RandomGenerator random);

    protected abstract double DoCalculate(double[] input);

    #endregion
  }

  /// <summary>
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight).
  /// FullNeuron stores whole array of weights even if there are always-zero weights (connetion is not exist).
  /// For a sparse-weighted neurons, please see SparseNeuron class
  /// </summary>
  public class FullNeuron : Neuron
  {
    private double[] m_Weights;

    public FullNeuron(int inputDim)
      : base(inputDim)
    {
      m_Weights = new double[m_InputDim];
    }

    public FullNeuron(NeuralLayer layer)
      : base(layer)
    {
      m_Weights = new double[m_InputDim];
    }


    /// <summary>
    /// Total count of existing connections between neuron and previous neural layer
    /// </summary>
    public override int ParamCount { get { return m_Weights.Length + 1; } }

    public override double this[int idx]
    {
      get { return m_Weights[idx]; }
      set { m_Weights[idx] = value; }
    }

    public override void AddSynapse(double weight = 0.0D)
    {
      var weights = new double[m_InputDim];

      Array.Copy(m_Weights, weights, m_InputDim-1);
      weights[m_InputDim-1] = weight;

      m_Weights = weights;
    }

    protected override void DoRandomize(RandomGenerator random)
    {
      for (int i=0; i<this.InputDim; i++)
        this[i] = 2 * random.GenerateUniform(0, 1) / m_InputDim;

      m_Bias = 2 * random.GenerateUniform(0, 1) / m_InputDim;
    }

    protected override double DoCalculate(double[] input)
    {
      var value = m_Bias;

      for (int i=0; i<m_InputDim; i++)
        value += m_Weights[i] * input[i];

      return value;
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

  /// <summary>
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight).
  /// Optimized for sparse data, i.e. when only some weights are non-zero.
  /// Empirical threshold value of non-zero weights is 15-20%
  /// (i.e. use SparseNeuron if only l.t. 15-20% of neuron weights are non-zero; othervise use FullNeuron)
  /// </summary>
  public class SparseNeuron : Neuron
  {
    private Dictionary<int, double> m_Weights;

    public SparseNeuron(int inputDim)
      : base(inputDim)
    {
      m_Weights = new Dictionary<int, double>();
    }

    public SparseNeuron(NeuralLayer layer)
      : base(layer)
    {
      m_Weights = new Dictionary<int, double>();
    }

    /// <summary>
    /// Total count of existing connections between neuron and previous neural layer
    /// </summary>
    public override int ParamCount { get { return m_Weights.Count + 1; } }

    public override double this[int idx]
    {
      get
      {
        if (idx < 0 || (idx > m_InputDim))
          throw new IndexOutOfRangeException();

        double result;
        if (m_Weights.TryGetValue(idx, out result))
          return result;

        return 0;
      }
      set
      {
        if (idx < 0 || (idx > m_InputDim))
          throw new IndexOutOfRangeException();

        m_Weights[idx] = value;
      }
    }

    public override void AddSynapse(double weight = 0.0D)
    {
      m_Weights[m_InputDim-1] = weight;
    }

    /// <summary>
    /// Removes connection
    /// </summary>
    /// <param name="idx">Index of neuron in the previous layer</param>
    public void RemoveWeight(int idx)
    {
      m_Weights.Remove(idx);
    }


    protected override void DoRandomize(RandomGenerator random)
    {
      foreach (var wdata in m_Weights)
        m_Weights[wdata.Key] = 2 * random.GenerateUniform(0, 1) / m_InputDim;

      m_Bias = 2 * random.GenerateUniform(0, 1) / m_InputDim;
    }

    protected override double DoCalculate(double[] input)
    {
      var value = m_Bias;
      var count = m_Weights.Count;

      foreach (var wdata in m_Weights)
        value += wdata.Value * input[wdata.Key];

      return value;
    }

    protected override double DoGetParam(int idx)
    {
      if (idx<m_Weights.Count)  return m_Weights.ElementAt(idx).Value;
      if (idx==m_Weights.Count) return m_Bias;

      throw new MLException("Index is out of range");
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      if (idx<m_Weights.Count)
      {
        var wdata = m_Weights.ElementAt(idx);
        if (isDelta)
          m_Weights[wdata.Key] += value;
        else
          m_Weights[wdata.Key] = value;
      }
      else if (idx==m_Weights.Count)
      {
        if (isDelta)
          m_Bias += value;
        else
          m_Bias = value;
      }
      else
        throw new MLException("Index is out of range");
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      var keys = new List<int>(m_Weights.Keys);
      if (isDelta)
      {
        foreach (var key in keys)
          m_Weights[key] += pars[cursor++];

        m_Bias += pars[cursor];
      }
      else
      {
        foreach (var key in keys)
          m_Weights[key] = pars[cursor++];

        m_Bias = pars[cursor];
      }
    }
  }
}
