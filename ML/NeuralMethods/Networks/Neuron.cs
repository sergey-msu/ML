using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Networks
{
  /// <summary>
  /// Represents McCulloch–Pitts artificial neuron: a node with activation function and a list of pairs (index, weight)
  /// </summary>
  public class Neuron : ComputingNode<double[], double>
  {
    private readonly NeuralLayer m_Layer;
    private Dictionary<int, double> m_Weights;
    private IFunction m_ActivationFunction;

    public Neuron()
    {
      m_Weights = new Dictionary<int, double>();
    }

    public Neuron(NeuralLayer layer) : this()
    {
      if (layer==null)
        throw new MLException("Neuron.ctor(layer=null)");

      m_Layer = layer;
    }


    /// <summary>
    /// Layer which is neuron belongs to
    /// </summary>
    public NeuralLayer Layer { get { return m_Layer; } }

    /// <summary>
    /// Total count of existing connections between neuron and previous neural layer
    /// </summary>
    public override int ParamCount { get { return m_Weights.Count; } }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get
      {
        if (m_ActivationFunction != null)
          return m_ActivationFunction;

        return m_Layer != null ?
               m_Layer.ActivationFunction ?? Registry.ActivationFunctions.Identity :
               Registry.ActivationFunctions.Identity;
      }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for connection weights. Set 'null' to remove connection at the given index
    /// </summary>
    public double? this[int idx]
    {
      get
      {
        double value;
        if (!m_Weights.TryGetValue(idx, out value)) return null;
        return value;
      }
      set
      {
        if (!value.HasValue) m_Weights.Remove(idx);
        else m_Weights[idx] = value.Value;
      }
    }

    /// <summary>
    /// Calculates result value produced by neuron
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double Calculate(double[] input)
    {
      var value = 0.0D;
      var count = m_Weights.Count;

      foreach (var wdata in m_Weights)
        value += wdata.Value * input[wdata.Key];

      return ActivationFunction.Value(value);
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
