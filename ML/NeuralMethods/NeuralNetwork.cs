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
  public partial class NeuralNetwork<TInput> where TInput : IFeaturable<double>
  {
    private NeuralLayer[] m_Layers;

    public NeuralNetwork()
    {
    }

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

    /// <summary>
    /// Creates new neural layer at the given position. Adds the result in the end if the position is not selected
    /// </summary>
    /// <param name="idx"></param>
    /// <returns></returns>
    public NeuralLayer CreateLayer(int idx = -1)
    {
      NeuralLayer layer;

      if (m_Layers == null)
      {
        if (idx>0)
          throw new MLException(string.Format("Unable to insert first layer at position '{0}'", idx));

        layer = new NeuralLayer(this);
        m_Layers = new NeuralLayer[1] { layer };
        return layer;
      }

      if (idx > m_Layers.Length)
          throw new MLException(string.Format("Unable to insert layer at position '{0}'", idx));

      layer = new NeuralLayer(this);
      var layers = new NeuralLayer[m_Layers.Length+1];
      if (idx < 0) idx = m_Layers.Length;

      for (int i=0; i<layers.Length; i++)
      {
        if (i<idx) layers[i] = m_Layers[i];
        else if (i==idx) layers[i] = layer;
        else layers[i] = m_Layers[i-1];
      }

      m_Layers = layers;

      return layer;
    }

    /// <summary>
    /// Removes layer from the network
    /// </summary>
    public bool RemoveLayer(NeuralLayer layer)
    {
      if (m_Layers == null || m_Layers.Length <= 0)
        return false;

      var idx = Array.IndexOf(m_Layers, layer);
      if (idx < 0) return false;

      return RemoveLayer(idx);
    }

    /// <summary>
    /// Removes layer at the given position from the network
    /// </summary>
    public bool RemoveLayer(int idx)
    {
      if (m_Layers == null || m_Layers.Length <= 0)
        return false;
      if (idx < 0 || idx >= m_Layers.Length)
        return false;

      var layers = new NeuralLayer[m_Layers.Length-1];
      for (int i=0; i<m_Layers.Length; i++)
      {
        if (i<idx)
          layers[i] = m_Layers[i];
        else if (i>idx)
          layers[i-1] = m_Layers[i];
      }

      m_Layers = layers;
      return true;
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
    public bool TrySetWeight(int layerIdx, int fromNeuronIndx, int neuronIndx, double? weight)
    {
      weight = null;
      var layer = this[layerIdx];
      if (layer==null) return false;

      var neuron = layer[neuronIndx];
      if (neuron == null) return false;

      neuron[fromNeuronIndx] = weight;
      return true;
    }

    public void CheckConsistency()
    {
      // TODO, returns consistency errors i.e. missing layers/neurons, broken connections etc.
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
    public double[] Calculate(TInput input)
    {
      if (m_Layers==null || m_Layers.Length <= 0)
        throw new MLException("Network contains no layers");

      var data = input.RawData;
      if (UseBias)
      {
        data = new double[input.RawData.Length+1];
        Array.Copy(input.RawData, data, input.RawData.Length);
        data[input.RawData.Length] = 1.0D;
      }

      var layerCount = m_Layers.Length;

      for (int i=0; i<layerCount; i++)
      {
        var layer = m_Layers[i];
        data = layer.Calculate(data);
      }

      return data;
    }
  }
}
