using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.NeuralMethods.Algorithms
{
  public class NeuralNetwork<TInput> where TInput : IFeatureContainer<double>
  {
    #region Inner

      public class Neuron
      {
        internal Neuron(NeuronLayer layer)
        {
          if (layer==null)
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
            if (m_ActivationFunction!=null)
              return m_ActivationFunction;

            return m_Layer.ActivationFunction;
          }
          set { m_ActivationFunction=value; }
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

      public class NeuronLayer
      {
        internal NeuronLayer(NeuralNetwork<TInput> network)
        {
          if (network==null)
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
            if (m_ActivationFunction!=null)
              return m_ActivationFunction;

            return m_Network.ActivationFunction;
          }
          set { m_ActivationFunction=value; }
        }


        public Neuron AddNeuron(int idx = -1)
        {
          Neuron neuron;

          if (m_Neurons == null)
          {
            if (idx>0)
              throw new MLException(string.Format("Unable to insert first nauron at position '{0}'", idx));

            neuron = new Neuron(this);
            m_Neurons = new Neuron[1] { neuron };
            return neuron;
          }

          if (idx > m_Neurons.Length)
              throw new MLException(string.Format("Unable to insert neuron at position '{0}'", idx));

          neuron = new Neuron(this);
          var neurons = new Neuron[m_Neurons.Length+1];
          if (idx < 0) idx = neurons.Length;

          for (int i=0; i<neurons.Length; i++)
          {
            if (i<idx) neurons[i] = m_Neurons[i];
            else if (i==idx) neurons[i] = neuron;
            else neurons[i] = m_Neurons[i-1];
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

          var neurons = new Neuron[m_Neurons.Length-1];
          for (int i=0; i<m_Neurons.Length; i++)
          {
            if (i<idx)
              neurons[i] = m_Neurons[i];
            else if (i>idx)
              neurons[i-1] = m_Neurons[i];
          }

          return true;
        }

        public double[] Calculate(double[] input)
        {
          if (m_Neurons==null || m_Neurons.Length <= 0)
            throw new MLException("Layer contains no neurons");

          var neuronCount = m_Neurons.Length;
          var output = new double[neuronCount];

          for (int i=0; i<neuronCount; i++)
          {
            var neuron = m_Neurons[i];
            output[i] = neuron.Calculate(input);
          }

          return output;
        }
      }

    #endregion

    public NeuralNetwork()
    {
    }

    private NeuronLayer[] m_Layers;

    public IFunction ActivationFunction { get; set; }
    public NeuronLayer[] Layers { get { return m_Layers; } }
    public NeuronLayer this[int i]
    {
      get { return m_Layers[i]; }
      set { m_Layers[i] = value; }
    }


    public NeuronLayer AddLayer(int idx = -1)
    {
      NeuronLayer layer;

      if (m_Layers == null)
      {
        if (idx>0)
          throw new MLException(string.Format("Unable to insert first layer at position '{0}'", idx));

        layer = new NeuronLayer(this);
        m_Layers = new NeuronLayer[1] { layer };
        return layer;
      }

      if (idx > m_Layers.Length)
          throw new MLException(string.Format("Unable to insert layer at position '{0}'", idx));

      layer = new NeuronLayer(this);
      var layers = new NeuronLayer[m_Layers.Length+1];
      if (idx < 0) idx = layers.Length;

      for (int i=0; i<layers.Length; i++)
      {
        if (i<idx) layers[i] = m_Layers[i];
        else if (i==idx) layers[i] = layer;
        else layers[i] = m_Layers[i-1];
      }

      m_Layers = layers;

      return layer;
    }

    public bool RemoveLayer(NeuronLayer layer)
    {
      if (m_Layers == null || m_Layers.Length <= 0)
        return false;

      var idx = Array.IndexOf(m_Layers, layer);
      if (idx < 0) return false;

      return RemoveLayer(idx);
    }

    public bool RemoveLayer(int idx)
    {
      if (m_Layers == null || m_Layers.Length <= 0)
        return false;
      if (idx < 0 || idx >= m_Layers.Length)
        return false;

      var layers = new NeuronLayer[m_Layers.Length-1];
      for (int i=0; i<m_Layers.Length; i++)
      {
        if (i<idx)
          layers[i] = m_Layers[i];
        else if (i>idx)
          layers[i-1] = m_Layers[i];
      }

      return true;
    }

    public double[] Calculate(TInput input)
    {
      if (m_Layers==null || m_Layers.Length <= 0)
        throw new MLException("Neural Netwok contains no layers");

      var data = input.RawData;
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
