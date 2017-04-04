using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Models
{
  /// <summary>
  /// Represents feedforward artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  public class NeuralNetwork : SequenceNode<double[], NeuralLayer>
  {
    #region Fields

    private IActivationFunction m_ActivationFunction;
    private bool m_IsTraining;
    private int m_InputDim;

    #endregion

    #region .ctor

    public NeuralNetwork(int inputDim, IActivationFunction activation = null)
    {
      if (inputDim <= 0)
        throw new MLException("NeuralLayer.ctor(inputDim<=0)");

      m_InputDim = inputDim;
      m_ActivationFunction = activation;
    }

    #endregion

    #region Properties

    /// <summary>
    /// True for using the network in training mode
    /// </summary>
    public bool IsTraining
    {
      get { return m_IsTraining; }
      set
      {
        m_IsTraining=value;
        foreach (var layer in SubNodes)
          layer.IsTraining = value;
      }
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
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for network layers
    /// </summary>
    public NeuralLayer this[int idx] { get { return SubNodes[idx]; } }

    #endregion

    #region Public

    /// <summary>
    /// Adds new neural layer in the end of layer list
    /// </summary>
    public virtual void AddLayer(NeuralLayer layer)
    {
      if (layer==null)
        throw new MLException("Layer can not be null");

      this.AddSubNode(layer);
    }

    /// <summary>
    /// Randomizes network weights
    /// </summary>
    public virtual void RandomizeParameters(int seed=0)
    {
      foreach (var layer in SubNodes)
        layer.RandomizeParameters(seed);
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

      var dim = InputDim;

      foreach (var layer in this.SubNodes)
      {
        layer.ActivationFunction = layer.ActivationFunction ?? ActivationFunction;
        layer.IsTraining = IsTraining;
        layer.m_InputDim = dim;
        layer.DoBuild();

        dim = layer.NeuronCount;
      }
    }

    #endregion

    #region Serialization

    public void Serialize(System.IO.Stream stream)
    {
      var serializer = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
      serializer.Serialize(stream, this);
    }

    public static NeuralNetwork Deserialize(System.IO.Stream stream)
    {
      var serializer = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
      return (NeuralNetwork)serializer.Deserialize(stream);
    }

    #endregion
  }
}
