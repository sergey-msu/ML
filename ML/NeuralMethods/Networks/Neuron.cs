using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Networks
{
  /// <summary>
  /// Represents artificial neuron: a node with activation function and a list of pairs (index, weight)
  /// </summary>
  public class Neuron // : OutputLayer<double[], double>
  {
    private readonly NeuralLayer m_Layer;
    private Dictionary<int, double> m_Weights;
    private IFunction m_ActivationFunction;

    internal Neuron(NeuralLayer layer)
    {
      if (layer == null)
        throw new MLException("Neuron.ctor(layer=null)");

      m_Layer = layer;
      m_Weights = new Dictionary<int, double>();
    }

    /// <summary>
    /// Layer which is neuron belongs to
    /// </summary>
    public NeuralLayer Layer { get { return m_Layer; } }

    /// <summary>
    /// Total count of existing connections between neuron and previous neural layer
    /// </summary>
    public int WeightCount { get { return m_Weights.Count; } }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
      {
        get
        {
          if (m_ActivationFunction != null)
            return m_ActivationFunction;

          return m_Layer.ActivationFunction;
        }
        set { m_ActivationFunction = value; }
      }

    /// <summary>
    /// Indexer for connection weights. Set 'null' to remove connection at the given index
    /// </summary>
    public double? this[int i]
      {
        get
        {
          double value;
          if (!m_Weights.TryGetValue(i, out value)) return null;
          return value;
        }
        set
        {
          if (!value.HasValue) m_Weights.Remove(i);
          else m_Weights[i] = value.Value;
        }
      }


    /// <summary>
      /// Updates neuron weights with the given array of values
      /// </summary>
      /// <param name="weights">Array of values</param>
      /// <param name="isDelta">Are the values in array is absolute or just deltas</param>
      /// <param name="cursor">'From' index in the array</param>
    public void UpdateWeights(double[] weights, bool isDelta, ref int cursor)
      {
        var idx = cursor;
        var keys = new List<int>(m_Weights.Keys);

        foreach (var key in keys)
        {
          if (idx >= weights.Length) break;

          var value = weights[idx];
          if (isDelta)
            m_Weights[key] += value;
          else
            m_Weights[key] = value;

          idx++;
        }

        cursor = idx;
      }

    /// <summary>
    /// Calculates result value produced by neuron
    /// </summary>
    /// <param name="input">Input data array</param>
    public double Calculate(double[] input)
    {
      var value = 0.0D;
      var count = m_Weights.Count;

      foreach (var wdata in m_Weights)
        value += wdata.Value * input[wdata.Key];

      return ActivationFunction.Value(value);
    }
  }
}
