using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Core.ComputingNetworks
{
  //public struct Indx
  //{
  //  public bool IsEmpty;
  //  public int Begin;
  //  public int End;
  //  public Indx[] SubIndxs;
  //}

  /// <summary>
  /// Represents Computing Layer as a linked list of other values that can accomplish some calculations
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  public interface IComputingLayer<TIn>
  {
    /// <summary>
    /// Calculates final calculation result.
    /// It may differ (for non-terminating layers) from direct layer rcalculation result:
    /// L = [L1 -> [L2 -> L3]]
    /// L(x) = L3(L2(L1(x))) - the final result, not L1(x)
    /// </summary>
    object Calculate(TIn input);

    /// <summary>
    /// Tries to update parameters of the network and passes it down to all sublayers
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

    /// <summary>
    /// Compiles Layer (build parameter index etc.)
    /// </summary>
    void Compile();
  }
}
