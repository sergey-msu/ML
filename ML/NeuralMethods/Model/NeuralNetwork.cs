using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core.ComputingNetworks;
using ML.Core;

namespace ML.NeuralMethods.Model
{
  /// <summary>
  /// Represents feedforward artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  public class NeuralNetwork : SequenceNode<double[], NeuralLayer>
  {
    public IFunction m_ActivationFunction;

    public NeuralNetwork()
    {
    }

    /// <summary>
    /// Total count of network layers (hidden + output)
    /// </summary>
    public int LayerCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get; set; }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for networl layers
    /// </summary>
    public NeuralLayer this[int idx] { get { return SubNodes[idx]; } }

    /// <summary>
    /// Creates new neural layer and adds the result in the end of layer list
    /// </summary>
    public virtual NeuralLayer CreateLayer()
    {
      var len = SubNodes.Length;
      var inputDim = (len <= 0) ? InputDim : SubNodes[len-1].SubNodes.Length;

      var layer = new NeuralLayer(this, inputDim);
      this.AddSubNode(layer);
      return layer;
    }

    /// <summary>
    /// Add neuron in the end of some layer.
    /// All neurons in the next layer will rearrange its weights correspondingly
    /// </summary>
    public void AddNeuron(Neuron neuron, int layerIdx)
    {
      if (neuron==null)
        throw new MLException("Neuron can not be null");
      if (layerIdx<0 || layerIdx>=LayerCount)
        throw new MLException("Wrong layer index");

      var layer = this[layerIdx];
      layer.AddNeuron(neuron);

      if (layerIdx < LayerCount-1)
      {
        var next = this[layerIdx+1];
        next.InputDim++;
        var ncount = next.NeuronCount;
        for (int i=0; i<ncount; i++)
          next[i].AddSynapse();
      }
    }

    /// <summary>
    /// Randomizes network weights
    /// </summary>
    public void Randomize(int seed=0)
    {
      foreach (var layer in SubNodes)
        layer.Randomize(seed);
    }

    /// <summary>
    /// Calculates result array produced by network
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[] Calculate(double[] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      return base.Calculate(input);
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      ActivationFunction = ActivationFunction ?? Registry.ActivationFunctions.Identity;

      base.DoBuild();
    }
  }
}
