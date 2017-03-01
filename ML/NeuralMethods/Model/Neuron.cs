using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Model
{
  /// <summary>
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight)
  /// </summary>
  public abstract class Neuron : ComputingNode<double[], double>
  {
    private readonly NeuralLayer m_Layer;
    private IFunction m_ActivationFunction;
    protected int  m_InputDim;
    protected bool m_UseBias;
    private double m_NetValue;
    private double m_Value;

    public Neuron(int inputDim, bool useBias)
    {
      if (inputDim <= 0)
        throw new MLException("Neuron.ctor(inputDim<=0)");

      m_InputDim = inputDim;
      m_UseBias = useBias;
    }

    public Neuron(NeuralLayer layer, int inputDim)
      : this(inputDim, layer.UseBias)
    {
      if (layer==null)
        throw new MLException("Neuron.ctor(layer=null)");

      m_Layer = layer;
    }


    /// <summary>
    /// Caclulated pure value (before applying activation function)
    /// </summary>
    public double NetValue { get { return m_NetValue; } }

    /// <summary>
    /// Caclulated value (after applying activation function)
    /// </summary>
    public double Value { get { return m_Value; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get { return m_InputDim; } }

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
    /// Whether or not use bias connection - net value shift
    /// </summary>
    public bool UseBias
    {
      get { return m_UseBias; }
    }

    /// <summary>
    /// Indexer for connection weights.
    /// Warning! idx here is an index of neuron in the previous layer NOT SERIAL (indexed) parameter index.
    /// Neuron may be connected with 0,1,4,7,8-th neurons in the previous layer,
    /// while through parameter index will be serial: 0,1,2,3,4
    /// (therefore one can not use DoGetParam/DoSetParam methods as they operate with serial indexed parameter indices).
    /// </summary>
    public abstract double this[int idx] { get; set; }

    /// <summary>
    /// Calculates result value produced by neuron
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double Calculate(double[] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorret input vector dimension");

      m_NetValue = DoCalculate(input);
      m_Value = m_ActivationFunction.Value(m_NetValue);

      return m_Value;
    }

    protected abstract double DoCalculate(double[] input);


    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      if (ActivationFunction==null)
        ActivationFunction = (Layer != null ? Layer.ActivationFunction : null) ?? Registry.ActivationFunctions.Identity;

      base.DoBuild();
    }
  }

  /// <summary>
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight).
  /// FullNeuron stores whole array of weights even if there are always-zero weights (connetion is not exist).
  /// For a sparse-weighted neurons, please see SparseNeuron class
  /// </summary>
  public class FullNeuron : Neuron
  {
    private double[] m_Weights;

    public FullNeuron(int inputDim, bool useBias)
      : base(inputDim, useBias)
    {
      var dim = m_UseBias ? inputDim+1 : inputDim;
      m_Weights = new double[dim];
    }

    public FullNeuron(NeuralLayer layer, int inputDim) : base(layer, inputDim)
    {
      var dim = m_UseBias ? inputDim+1 : inputDim;
      m_Weights = new double[dim];
    }


    /// <summary>
    /// Total count of existing connections between neuron and previous neural layer
    /// </summary>
    public override int ParamCount { get { return m_Weights.Length; } }

    public override double this[int idx]
    {
      get
      {
        if (idx < 0 || (idx > m_InputDim) || (!m_UseBias && idx == m_InputDim))
          throw new IndexOutOfRangeException();

        return m_Weights[idx];
      }
      set
      {
        if (idx < 0 || (idx > m_InputDim) || (!m_UseBias && idx == m_InputDim))
          throw new IndexOutOfRangeException();

        m_Weights[idx] = value;
      }
    }


    protected override double DoCalculate(double[] input)
    {
      var value = 0.0D;
      var len = input.Length;

      for (int i=0; i<len; i++)
        value += m_Weights[i] * input[i];

      if (m_UseBias) value += m_Weights[len]; // bias weight is the last element of weight array (by definition)

      return value;
    }

    protected override double DoGetParam(int idx)
    {
      return m_Weights[idx];
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      if (isDelta)
        m_Weights[idx] += value;
      else
        m_Weights[idx] = value;
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      var len = m_Weights.Length;
      if (isDelta)
      {
        for (int i=0; i<len; i++)
          m_Weights[i] += pars[cursor+i];
      }
      else
      {
        Array.Copy(pars, cursor, m_Weights, 0, len);
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

    public SparseNeuron(int inputDim, bool useBias)
      : base(inputDim, useBias)
    {
      m_Weights = new Dictionary<int, double>();
    }

    public SparseNeuron(NeuralLayer layer, int inputDim)
      : base(layer, inputDim)
    {
      m_Weights = new Dictionary<int, double>();
    }

    /// <summary>
    /// Total count of existing connections between neuron and previous neural layer
    /// </summary>
    public override int ParamCount { get { return m_Weights.Count; } }

    public override double this[int idx]
    {
      get
      {
        if (idx < 0 || (idx > m_InputDim) || (!m_UseBias && idx == m_InputDim))
          throw new IndexOutOfRangeException();

        double result;
        if (m_Weights.TryGetValue(idx, out result))
          return result;

        return 0;
      }
      set
      {
        if (idx < 0 || (idx > m_InputDim) || (!m_UseBias && idx == m_InputDim))
          throw new IndexOutOfRangeException();

        m_Weights[idx] = value;;
      }
    }

    /// <summary>
    /// Removes connection
    /// </summary>
    /// <param name="idx">Index of neuron in the previous layer</param>
    public void RemoveWeight(int idx)
    {
      m_Weights.Remove(idx);
    }


    protected override double DoCalculate(double[] input)
    {
      var value = 0.0D;
      var count = m_Weights.Count;

      foreach (var wdata in m_Weights)
      {
        if (wdata.Key == input.Length) // bias weight is the last element of weight array (by definition)
          value += wdata.Value;
        else
          value += wdata.Value * input[wdata.Key];
      }

      return value;
    }

    protected override double DoGetParam(int idx)
    {
      return m_Weights.ElementAt(idx).Value;
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      var wdata = m_Weights.ElementAt(idx);
      if (isDelta)
        m_Weights[wdata.Key] += value;
      else
        m_Weights[wdata.Key] = value;
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      var keys = new List<int>(m_Weights.Keys);
      if (isDelta)
      {
        foreach (var key in keys)
          m_Weights[key] += pars[cursor++];
      }
      else
      {
        foreach (var key in keys)
          m_Weights[key] = pars[cursor++];
      }
    }
  }
}
