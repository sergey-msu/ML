using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.NeuralMethods
{
  public partial class NeuralNetwork<TInput> where TInput : IFeatureContainer<double>
  {
    public class NeuralLayer
    {
      internal NeuralLayer(NeuralNetwork<TInput> network)
      {
        if (network == null)
          throw new MLException("NeuronLayer.ctor(network=null)");

        m_Network = network;
      }

      private readonly NeuralNetwork<TInput> m_Network;
      private Neuron[] m_Neurons;
      private IFunction m_ActivationFunction;

      public NeuralNetwork<TInput> Network { get { return m_Network; } }
      public Neuron[] Neurons { get { return m_Neurons; } }
      public Neuron this[int i]
      {
        get { return m_Neurons[i]; }
        set { m_Neurons[i] = value; }
      }
      public IFunction ActivationFunction
      {
        get
        {
          if (m_ActivationFunction != null)
            return m_ActivationFunction;

          return m_Network.ActivationFunction;
        }
        set { m_ActivationFunction = value; }
      }


      public Neuron AddNeuron(int idx = -1)
      {
        Neuron neuron;

        if (m_Neurons == null)
        {
          if (idx > 0)
            throw new MLException(string.Format("Unable to insert first nauron at position '{0}'", idx));

          neuron = new Neuron(this);
          m_Neurons = new Neuron[1] { neuron };
          return neuron;
        }

        if (idx > m_Neurons.Length)
          throw new MLException(string.Format("Unable to insert neuron at position '{0}'", idx));

        neuron = new Neuron(this);
        var neurons = new Neuron[m_Neurons.Length + 1];
        if (idx < 0) idx = m_Neurons.Length;

        for (int i = 0; i < neurons.Length; i++)
        {
          if (i < idx) neurons[i] = m_Neurons[i];
          else if (i == idx) neurons[i] = neuron;
          else neurons[i] = m_Neurons[i - 1];
        }

        m_Neurons = neurons;

        return neuron;
      }

      public bool RemoveNeuron(Neuron neuron)
      {
        if (m_Neurons == null || m_Neurons.Length <= 0)
          return false;

        var idx = Array.IndexOf(m_Neurons, neuron);
        if (idx < 0) return false;

        return RemoveNeuron(idx);
      }

      public bool RemoveNeuron(int idx)
      {
        if (m_Neurons == null || m_Neurons.Length <= 0)
          return false;
        if (idx < 0 || idx >= m_Neurons.Length)
          return false;

        var neurons = new Neuron[m_Neurons.Length - 1];
        for (int i = 0; i < m_Neurons.Length; i++)
        {
          if (i < idx)
            neurons[i] = m_Neurons[i];
          else if (i > idx)
            neurons[i - 1] = m_Neurons[i];
        }

        return true;
      }

      public NeuralLayer PrevLayer()
      {
        var idx = Array.IndexOf(m_Network.Layers, this);
        if (idx <= 0) return null;
        return m_Network.Layers[idx-1];
      }

      public NeuralLayer NextLayer()
      {
        var idx = Array.IndexOf(m_Network.Layers, this);
        if (idx >= m_Network.Layers.Length-1) return null;
        return m_Network.Layers[idx+1];
      }

      public void UpdateWeights(double[] weights, ref int cursor)
      {
        var neuronCount = m_Neurons.Length;
        for (int i=0; i<neuronCount; i++)
        {
          if (cursor >= weights.Length) break;

          var neuron = m_Neurons[i];
          neuron.UpdateWeights(weights, ref cursor);
        }
      }

      public double[] Calculate(double[] input)
      {
        if (m_Neurons == null || m_Neurons.Length <= 0)
          throw new MLException("Layer contains no neurons");

        var neuronCount = m_Neurons.Length;
        var output = new double[neuronCount];

        for (int i = 0; i < neuronCount; i++)
        {
          var neuron = m_Neurons[i];
          output[i] = neuron.Calculate(input);
        }

        return output;
      }
    }
  }
}
