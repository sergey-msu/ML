using System;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Contract for a computing node as a black box that can accomplish some calculations
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public interface IComputingNode<TIn, TOut>
  {
    /// <summary>
    /// Returns number of node parameters
    /// </summary>
    int ParamCount { get; }

    /// <summary>
    /// Calculates node result
    /// </summary>
    TOut Calculate(TIn input);

    /// <summary>
    /// Builds node before use (build search index etc)
    /// </summary>
    void Build();

    /// <summary>
    /// Tries to update node parameters
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor);

    /// <summary>
    /// Tries to set parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    bool TrySetParam(int idx, double value, bool isDelta);

    /// <summary>
    /// Tries to return parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    bool TryGetParam(int idx, out double value);
  }

}
