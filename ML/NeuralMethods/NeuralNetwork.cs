using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Contracts;

namespace ML.NeuralMethods
{
  public partial class NeuralNetwork<TInput> where TInput : IFeatureContainer<double>
  {
    public NeuralNetwork()
    {
    }

    private NeuralLayer[] m_Layers;

    public IFunction ActivationFunction { get; set; }
    public NeuralLayer[] Layers { get { return m_Layers; } }
    public NeuralLayer this[int i]
    {
      get { return m_Layers[i]; }
      set { m_Layers[i] = value; }
    }


    public NeuralLayer AddLayer(int idx = -1)
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

    public bool RemoveLayer(NeuralLayer layer)
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

      var layers = new NeuralLayer[m_Layers.Length-1];
      for (int i=0; i<m_Layers.Length; i++)
      {
        if (i<idx)
          layers[i] = m_Layers[i];
        else if (i>idx)
          layers[i-1] = m_Layers[i];
      }

      return true;
    }

    public void UpdateWeights(double[] weights)
    {
      if (weights==null)
        throw new MLException("Weights are null");

      var layerCount = m_Layers.Length;
      var cursor = 0;

      for (int i=0; i<layerCount; i++)
      {
        if (cursor >= weights.Length) break;

        var layer = m_Layers[i];
        layer.UpdateWeights(weights, ref cursor);
      }
    }

    public double[] Calculate(TInput input)
    {
      if (m_Layers==null || m_Layers.Length <= 0)
        throw new MLException("Network contains no layers");

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
