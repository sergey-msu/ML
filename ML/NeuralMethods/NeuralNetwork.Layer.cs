using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.NeuralMethods
{
  public partial class NeuralNetwork
  {
    /// <summary>
    /// Represents artificial neural layer as a list of neurons
    /// </summary>
    public class NeuralLayer
    {
      private readonly NeuralNetwork m_Network;
      private Neuron[] m_Neurons;
      private IFunction m_ActivationFunction;

      internal NeuralLayer(NeuralNetwork network)
      {
        if (network == null)
          throw new MLException("NeuronLayer.ctor(network=null)");

        m_Network = network;
      }

      /// <summary>
      /// If true, adds artificial +1 input value in the and of input data array
      /// </summary>
      public bool UseBias { get; set; }

      /// <summary>
      /// In true, norms output vector to the summ value of it's components (give more 'probabalistic' meaning to the result)
      /// </summary>
      public bool NormOutput { get; set; }

      /// <summary>
      /// Neural network that the layer belongs to
      /// </summary>
      public NeuralNetwork Network { get { return m_Network; } }

      /// <summary>
      /// A list of the layer neurons
      /// </summary>
      public Neuron[] Neurons { get { return m_Neurons; } }

      /// <summary>
      /// Indexer for layer neurons
      /// </summary>
      public Neuron this[int i]
      {
        get
        {
          var count = m_Neurons.Length;
          return (i>=0 && i<count) ? m_Neurons[i] : null;
        }
      }

      /// <summary>
      /// Layer activation function. If null, the network's activation function will be used
      /// </summary>
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


      /// <summary>
      /// Creates new neuron at the given position. Adds the result in the end if the position is not selected
      /// </summary>
      public Neuron CreateNeuron(int idx = -1)
      {
        Neuron neuron;

        if (m_Neurons == null)
        {
          if (idx > 0)
            throw new MLException(string.Format("Unable to insert first neuron at position '{0}'", idx));

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

      /// <summary>
      /// Removes neuron from the layer
      /// </summary>
      public bool RemoveNeuron(Neuron neuron)
      {
        if (m_Neurons == null || m_Neurons.Length <= 0)
          return false;

        var idx = Array.IndexOf(m_Neurons, neuron);
        if (idx < 0) return false;

        return RemoveNeuron(idx);
      }

      /// <summary>
      /// Removes neuron at the given position from the layer
      /// </summary>
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

        m_Neurons = neurons;
        return true;
      }

      /// <summary>
      /// Updates all layer neurons weights with the given array of values
      /// </summary>
      /// <param name="weights">Array of values</param>
      /// <param name="isDelta">Are the values in array is absolute or just deltas</param>
      /// <param name="cursor">'From' index in the array</param>
      public void UpdateWeights(double[] weights, bool isDelta, ref int cursor)
      {
        var neuronCount = m_Neurons.Length;
        for (int i=0; i<neuronCount; i++)
        {
          if (cursor >= weights.Length) break;

          var neuron = m_Neurons[i];
          neuron.UpdateWeights(weights, isDelta, ref cursor);
        }
      }

      /// <summary>
      /// Calculates result array produced by layer
      /// </summary>
      /// <param name="input">Input data array</param>
      public double[] Calculate(double[] input)
      {
        if (m_Neurons == null || m_Neurons.Length <= 0)
          throw new MLException("Layer contains no neurons");

        var neuronCount = m_Neurons.Length;
        var output = new double[neuronCount];

        var data = input;
        if (UseBias)
        {
          data = new double[input.Length+1];
          Array.Copy(input, data, input.Length);
          data[input.Length] = 1.0D;
        }

        var sum = 0.0D;

        for (int i = 0; i < neuronCount; i++)
        {
          var neuron = m_Neurons[i];
          var calc = neuron.Calculate(data);
          sum += Math.Abs(calc);
          output[i] = calc;
        }

        if (NormOutput)
        {
          for (int i=0; i<neuronCount; i++)
            output[i] /= sum;
        }

        return output;
      }
    }
  }
}
