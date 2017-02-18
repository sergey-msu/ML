using System;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents abstract computing network as a linked list of computing layers
  /// NET -> [L1 -> [L2 -> [L3 -> L4]]]
  /// Each layer can aggregate another layers and networks, thus one can implement computing networks of any complexity
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public class ComputingNetwork<TIn, TOut> : IComputingLayer<TIn>
  {
    private IComputingLayer<TIn> m_BaseLayer;

    public ComputingNetwork()
    {
    }


    /// <summary>
    /// Returns number of network parameters
    /// </summary>
    public int ParamsCount { get { return 0; } }

    /// <summary>
    /// Returns network first(root) layer
    /// </summary>
    public IComputingLayer<TIn> BaseLayer { get { return m_BaseLayer; } }

    /// <summary>
    /// Adds Network root layer
    /// </summary>
    public void AddBaseLayer(IComputingLayer<TIn> layer)
    {
      m_BaseLayer = layer;
    }

    /// <summary>
    /// Passes input through linked list of sublayers and returns strong typed result
    /// </summary>
    public virtual TOut Calculate(TIn input)
    {
      if (m_BaseLayer==null)
        throw new MLException("Base layer has not been set");

      return (TOut)m_BaseLayer.Calculate(input);
    }

    /// <summary>
    /// Passes input through linked list of sublayers and returns the result
    /// </summary>
    object IComputingLayer<TIn>.Calculate(TIn input)
    {
      return Calculate(input);
    }

    /// <summary>
    /// Builds layer before use (build search index etc)
    /// </summary>
    public virtual void Build(bool buildIndex=true)
    {
      if (m_BaseLayer==null)
        throw new MLException("Base layer has not been set");

      m_BaseLayer.Build(buildIndex);
    }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>End index</returns>
    public virtual int BuildIndex(int startIdx)
    {
      throw new MLException("Base Computing network has no parameters and does not supports index");
    }

    /// <summary>
    /// Updates parameters of the network and passes it down to all sublayers
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public virtual bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (m_BaseLayer==null)
        throw new MLException("Base layer has not been set");

      return m_BaseLayer.TryUpdateParams(pars, isDelta, ref cursor);
    }

    /// <summary>
    /// Tries to return parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public virtual bool TryGetParam(int idx, out double value)
    {
      if (m_BaseLayer==null)
        throw new MLException("Base layer has not been set");

      return m_BaseLayer.TryGetParam(idx, out value);
    }

    /// <summary>
    /// Tries to set parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public bool TrySetParam(int idx, double value, bool isDelta)
    {
      if (m_BaseLayer==null)
        throw new MLException("Base layer has not been set");

      return m_BaseLayer.TrySetParam(idx, value, isDelta);
    }
  }

}
