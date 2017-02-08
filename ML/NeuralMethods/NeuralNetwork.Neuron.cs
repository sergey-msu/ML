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
      internal Neuron(NeuronLayer layer)
      {
        if (layer == null)
          throw new MLException("Neuron.ctor(layer=null)");

        m_Layer = layer;
      }

      private readonly NeuronLayer m_Layer;
      private IFunction m_ActivationFunction;

      public NeuronLayer Layer { get { return m_Layer; } }
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


      public double Calculate(double[] input)
      {
        var value = 0.0D;
        //for (int k=0; k<input.Length; k++)
        //{
        //  // if w.Length <= k then break;
        //  value += w[k]*data[k];
        //}

        return ActivationFunction.Calculate(value);
      }
    }
  }
}
