using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents abstract computing network as a linked list of computing layers
  /// NET -> [L1 -> [L2 -> [L3 -> L4]]]
  /// Each layer can aggregate another layers and networks, thus one can implement computing networks of any complexity
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public abstract class ComputingNetwork<TIn, TOut> : IComputingLayer<TIn>
  {
    private IComputingLayer<TIn> m_FirstLayer;

    public ComputingNetwork()
    {
    }

    /// <summary>
    /// Adds Network root layer
    /// </summary>
    public void AddFirstLayer(IComputingLayer<TIn> layer)
    {
      m_FirstLayer = layer;
    }

    /// <summary>
    /// Passes input through linked list of sublayers and returns strong typed result
    /// </summary>
    public TOut Calculate(TIn input)
    {
      return (TOut)m_FirstLayer.Calculate(input);
    }

    /// <summary>
    /// Passes input through linked list of sublayers and returns the result
    /// </summary>
    object IComputingLayer<TIn>.Calculate(TIn input)
    {
      return Calculate(input);
    }

    /// <summary>
    /// Updates parameters of the network and passes it down to all sublayers
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return m_FirstLayer.TryUpdateParams(pars, isDelta, ref cursor);
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
      return m_FirstLayer.TrySetParam(idx, value, isDelta);
    }

    /// <summary>
    /// Tries to return parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public bool TryGetParam(int idx, out double value)
    {
      return m_FirstLayer.TryGetParam(idx, out value);
    }

    /// <summary>
    /// Compiles Network (build parameter index etc.)
    /// </summary>
    public void Compile()
    {
      m_FirstLayer.Compile();
    }
  }

}
