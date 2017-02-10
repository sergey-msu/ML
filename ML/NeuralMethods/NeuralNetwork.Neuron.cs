using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;

namespace ML.NeuralMethods
{
  public partial class NeuralNetwork<TInput> where TInput : IFeatureContainer<double>
  {
    public class Neuron
    {
      internal Neuron(NeuralLayer layer)
      {
        if (layer == null)
          throw new MLException("Neuron.ctor(layer=null)");

        m_Layer = layer;
        m_Weights = new Dictionary<int, double>();
      }

      private readonly NeuralLayer m_Layer;
      private Dictionary<int, double> m_Weights;
      private IFunction m_ActivationFunction;

      public NeuralLayer Layer { get { return m_Layer; } }
      public int WeightCount { get { return m_Weights.Count; } }
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

      public double this[int idx]
      {
        get { return m_Weights[idx]; }
        set { m_Weights[idx] = value; }
      }

      public void UpdateWeights(double[] weights, bool isDelta, ref int cursor)
      {
        var idx = cursor;

        foreach (var wdata in m_Weights)
        {
          if (idx >= weights.Length) break;

          var value = weights[idx];
          if (isDelta)
            m_Weights[wdata.Key] += value;
          else
            m_Weights[wdata.Key] = value;

          idx++;
        }

        cursor = idx;
      }

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
}
