using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.NeuralMethods
{
  /// <summary>
  /// Represents artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  /// <typeparam name="TInput">Input data type</typeparam>
  public class NeuralNetwork
  {
    private NeuralLayer[] m_Layers;

    public NeuralNetwork()
    {
    }

    #region Properties

    /// <summary>
    /// If true, adds artificial +1 input value in the and of input data array
    /// </summary>
    public bool UseBias { get; set; }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IFunction ActivationFunction { get; set; }

    /// <summary>
    /// A list of network layers
    /// </summary>
    public NeuralLayer[] Layers { get { return m_Layers; } }

    /// <summary>
    /// Indexer for netwirk layers
    /// </summary>
    public NeuralLayer this[int i]
    {
      get
      {
        var count = m_Layers.Length;
        return (i>=0 && i<count) ? m_Layers[i] : null;
      }
    }

    #endregion

    #region Public

    /// <summary>
    /// Creates new neural layer. Adds the result in the end
    /// </summary>
    public NeuralLayer CreateLayer()
    {
      var layer = new NeuralLayer(this);

      if (m_Layers == null)
      {
        m_Layers = new NeuralLayer[1] { layer };
        return layer;
      }

      var layers = new NeuralLayer[m_Layers.Length+1];

      for (int i=0; i<m_Layers.Length; i++)
        layers[i] = m_Layers[i];
      layers[m_Layers.Length] = layer;

      m_Layers = layers;

      return layer;
    }

    /// <summary>
    /// Removes last layer from the network
    /// </summary>
    public NeuralLayer RemoveLayer()
    {
      if (m_Layers == null || m_Layers.Length <= 0)
        return null;

      var idx = m_Layers.Length-1;
      var layer = m_Layers[idx];

      var layers = new NeuralLayer[idx];
      for (int i = 0; i < idx; i++)
        layers[i] = m_Layers[i];

      m_Layers = layers;

      return layer;
    }

    public void CheckConsistency()
    {
      // TODO, returns consistency errors i.e. missing layers/neurons, broken connections etc.
    }

    /// <summary>
    /// Tries to get weight of the connection between neurons in given layer
    /// </summary>
    /// <param name="layerIdx">Index of the layer</param>
    /// <param name="fromNeuronIndx">Index of the 'from' neuron from the previous layer</param>
    /// <param name="neuronIndx">Index of the 'to' neuron in the given layer</param>
    /// <returns>Returns false if layer or neuron does not exist</returns>
    public bool TryGetWeight(int layerIdx, int fromNeuronIndx, int neuronIndx, out double? weight)
    {
      weight = null;
      var layer = this[layerIdx];
      if (layer==null) return false;

      var neuron = layer[neuronIndx];
      if (neuron == null) return false;

      weight = neuron[fromNeuronIndx];
      return true;
    }

    /// <summary>
    /// Tries to set weight value of the connection between neurons in given layer
    /// </summary>
    /// <param name="layerIdx">Index of the layer</param>
    /// <param name="fromNeuronIndx">Index of the 'from' neuron from the previous layer</param>
    /// <param name="neuronIndx">Index of the 'to' neuron in the given layer</param>
    /// <returns>Returns false if layer or neuron does not exist</returns>
    public bool TrySetWeight(int layerIdx, int fromNeuronIndx, int neuronIndx, double? weight, bool isDelta)
    {
      var layer = this[layerIdx];
      if (layer==null) return false;

      var neuron = layer[neuronIndx];
      if (neuron == null) return false;

      if (isDelta)
        neuron[fromNeuronIndx] += weight;
      else
        neuron[fromNeuronIndx] = weight;

      return true;
    }

    /// <summary>
    /// Updates all layer neurons weights with the given array of values
    /// </summary>
    /// <param name="weights">Array of values</param>
    /// <param name="isDelta">Are the values in array is absolute or just deltas</param>
    /// <param name="cursor">'From' index in the array</param>
    public void UpdateWeights(double[] weights, bool isDelta)
    {
      if (weights==null)
        throw new MLException("Weights are null");

      var layerCount = m_Layers.Length;
      var cursor = 0;

      for (int i=0; i<layerCount; i++)
      {
        if (cursor >= weights.Length) break;

        var layer = m_Layers[i];
        layer.UpdateWeights(weights, isDelta, ref cursor);
      }
    }

    /// <summary>
    /// Calculates result array produced by network
    /// </summary>
    /// <param name="input">Input data array</param>
    public virtual double[] Calculate(double[] input)
    {
      if (m_Layers==null || m_Layers.Length <= 0)
        throw new MLException("Network contains no layers");

      double[] data = input;
      if (UseBias)
      {
        data = new double[input.Length+1];
        Array.Copy(input, data, input.Length);
        data[input.Length] = 1.0D;
      }

      var layerCount = m_Layers.Length;

      for (int i=0; i<layerCount; i++)
      {
        var layer = m_Layers[i];
        data = layer.Calculate(data);
      }

      return data;
    }

    #endregion
  }
}
