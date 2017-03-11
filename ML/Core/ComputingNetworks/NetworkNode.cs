using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents feedforward artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  public abstract class NetworkNode<TPar, TLayer, TNeuron> : SequenceNode<TPar[], TLayer>
    where TLayer  : LayerNode<TPar, TNeuron>
    where TNeuron : NeuronNode<TPar>
  {
    private int m_InputDim;
    private IFunction m_ActivationFunction;

    protected NetworkNode(int inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("NetworkNode.ctor(inputDim<=0)");

      m_InputDim = inputDim;
    }

    /// <summary>
    /// Total count of network layers (hidden + output)
    /// </summary>
    public int LayerCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get { return m_InputDim; } }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for network layers
    /// </summary>
    public TLayer this[int idx] { get { return SubNodes[idx]; } }

    /// <summary>
    /// Adds new neural layer in the end of layer list
    /// </summary>
    public virtual void AddLayer(TLayer layer)
    {
      if (layer==null)
        throw new MLException("Layer can not be null");

      var prevOutputDim = (LayerCount == 0) ? InputDim : this[LayerCount-1].NeuronCount;
      if (layer.InputDim != prevOutputDim)
        throw new MLException("Layer input dimension differs with layer's one");

      this.AddSubNode(layer);
    }

    /// <summary>
    /// Randomizes network weights
    /// </summary>
    public void RandomizeParameters(int seed=0)
    {
      foreach (var layer in SubNodes)
        layer.RandomizeParameters(seed);
    }

    /// <summary>
    /// Calculates result array produced by network
    /// </summary>
    /// <param name="input">Input data array</param>
    public override TPar[] Calculate(TPar[] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      return base.Calculate(input);
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      foreach (var layer in this.SubNodes)
        layer.ActivationFunction = layer.ActivationFunction ?? ActivationFunction;

      base.DoBuild();
    }
  }
}
