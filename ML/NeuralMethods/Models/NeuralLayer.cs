using System;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Models
{
  /// <summary>
  /// Represents artificial neural layer as a list of neurons
  /// </summary>
  public class NeuralLayer : CompositeNode<double[], double, Neuron>
  {
    #region Fields

    private IActivationFunction m_ActivationFunction;
    private bool m_IsTraining;
    private double m_DropoutRate;
    private int m_DropoutSeed;
    internal int m_InputDim;

    #endregion

    #region .ctor

    public NeuralLayer(int neuronCount, IActivationFunction actiovation = null)
    {
      if (neuronCount <= 0)
        throw new MLException("NeuralLayer.ctor(neuronCount<=0)");

      m_ActivationFunction = actiovation;

      for (int i=0; i<neuronCount; i++)
      {
        var neuron = new Neuron();
        AddNeuron(neuron);
      }
    }

    #endregion

    #region Properties

    /// <summary>
    /// If true, indicates that layer is in training mode
    /// </summary>
    public bool IsTraining
    {
      get { return m_IsTraining; }
      set
      {
        m_IsTraining=value;
        foreach (var neuron in SubNodes)
          neuron.IsTraining = value;
      }
    }

    public double DropoutRate
    {
      get { return m_DropoutRate; }
      set
      {
        m_DropoutRate=value;
        foreach (var neuron in SubNodes)
          neuron.DropoutRate = value;
      }
    }


    /// <summary>
    /// Random seed for dropout
    /// </summary>
    public int DropoutSeed
    {
      get { return m_DropoutSeed; }
      set
      {
        m_DropoutSeed=value;
        foreach (var neuron in SubNodes)
          neuron.DropoutSeed = value;
      }
    }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get { return m_InputDim; } }

    /// <summary>
    /// Total count of abstract neurons (i.e. neurons in NN, feature maps in CNN etc.)
    /// </summary>
    public int NeuronCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for layer abstract neurons (i.e. neurons in NN, feature maps in CNN etc.)
    /// </summary>
    public Neuron this[int idx] { get { return SubNodes[idx]; } }

    #endregion

    #region Public

    /// <summary>
    /// Add existing abstract neurons (i.e. neurons in NN, feature maps in CNN etc.) in the end of the layer
    /// </summary>
    public virtual void AddNeuron(Neuron neuron)
    {
      if (neuron==null)
        throw new MLException("Neuron can not be null");

      this.AddSubNode(neuron);
    }

    /// <summary>
    /// Randomizes layer neurons parameters (weights for NNs)
    /// </summary>
    public virtual void RandomizeParameters(int seed=0)
    {
      foreach (var neuron in this.SubNodes)
        neuron.RandomizeParameters(seed);
    }

    /// <summary>
    /// Calculates result array produced by layer
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[] Calculate(double[] input)
    {
      if (m_InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      return base.Calculate(input);
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      foreach (var neuron in this.SubNodes)
      {
        neuron.ActivationFunction = neuron.ActivationFunction ?? ActivationFunction;
        neuron.IsTraining = IsTraining;
        if (m_DropoutRate>0) neuron.DropoutRate = DropoutRate;
        neuron.m_InputDim = m_InputDim;

        neuron.DoBuild();
      }
    }

    #endregion
  }
}
