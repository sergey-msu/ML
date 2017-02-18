using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.NeuralMethods
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
    /// In true, norms output vector to the summ of abs values of it's components (give more 'probabalistic' meaning to the result)
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
    /// Creates new neuron. Adds the result in the end
    /// </summary>
    public Neuron CreateNeuron()
    {
      var neuron = new Neuron(this);

      if (m_Neurons == null)
      {
        m_Neurons = new Neuron[1] { neuron };
        return neuron;
      }

      var neurons = new Neuron[m_Neurons.Length + 1];

      for (int i=0; i<m_Neurons.Length; i++)
        neurons[i] = m_Neurons[i];
      neurons[m_Neurons.Length] = neuron;

      m_Neurons = neurons;

      return neuron;
    }

    /// <summary>
    /// Removes last neuron from the layer
    /// </summary>
    public Neuron RemoveNeuron()
    {
      if (m_Neurons == null || m_Neurons.Length <= 0)
        return null;

      var idx = m_Neurons.Length-1;
      var neuron = m_Neurons[idx];
      var nextLayer = NextLayer();
      if (nextLayer != null)
      {
        for (int i=0; i<nextLayer.Neurons.Length; i++)
          neuron[idx] = null;
      }

      var neurons = new Neuron[idx];
      for (int i = 0; i < idx; i++)
        neurons[i] = m_Neurons[i];

      m_Neurons = neurons;

      return neuron;
    }

    /// <summary>
    /// Returns layer previous to this in the network
    /// </summary>
    public NeuralLayer PrevLayer()
    {
      var idx = Array.IndexOf(Network.Layers, this);
      if (idx <= 0) return null;

      return Network[idx-1];
    }

    /// <summary>
    /// Returns layer next to this in the network
    /// </summary>
    public NeuralLayer NextLayer()
    {
      var idx = Array.IndexOf(Network.Layers, this);
      if (idx >= Network.Layers.Length-1) return null;

      return Network[idx+1];
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
    public virtual double[] Calculate(double[] input)
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
