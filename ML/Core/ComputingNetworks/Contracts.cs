using System;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Index for fast search parameter within layer
  /// </summary>
  internal struct ParamIdx
  {
    public ParamIdx(int start, int self, int end)
    {
      if (start > self || self > end)
        throw new MLCorruptedIndexException();

      IsNotEmpty = true;
      Start = start;
      Self = self;
      End = end;
    }

    public bool IsNotEmpty;
    public int Start;
    public int Self;
    public int End;

    public bool CheckSelf(int idx)
    {
      return IsNotEmpty && (Start <= idx) && (idx < Self);
    }

    public bool CheckEnd(int idx)
    {
      return IsNotEmpty && (Self <= idx) && (idx < End);
    }

    public void MoveCursor(ref int cursor)
    {
      cursor += (Self-Start); // = layer.ParamsCount
    }
  }

  /// <summary>
  /// Contract for a Computing Layer as a linked list of other values that can accomplish some calculations
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  public interface IComputingLayer<TIn>
  {
    /// <summary>
    /// Returns number of layer parameters
    /// </summary>
    int ParamsCount { get; }

    /// <summary>
    /// Calculates final calculation result.
    /// It may differ (for non-terminating layers) from direct layer rcalculation result:
    /// L = [L1 -> [L2 -> L3]]
    /// L(x) = L3(L2(L1(x))) - the final result, not L1(x)
    /// </summary>
    object Calculate(TIn input);

    /// <summary>
    /// Builds layer before use (build search index etc)
    /// </summary>
    void Build(bool buildIndex=true);

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>End index</returns>
    int BuildIndex(int startIdx);

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
  }

  /// <summary>
  /// Contract for a terminating layer that has no next layers
  /// </summary>
  /// <typeparam name="TIn"></typeparam>
  /// <typeparam name="TOut"></typeparam>
  public interface IOutputLayer<TIn, TOut> : IComputingLayer<TIn>
  {
  }

  /// <summary>
  /// Contract for a hidden computing layer -
  /// non-terminating layer that calculates some result and passes it further in the layer chain
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Direct output object type</typeparam>
  public interface IHiddenLayer<TIn, TOut> : IComputingLayer<TIn>
  {
    /// <summary>
    /// Next layer in layer chain
    /// </summary>
    IComputingLayer<TOut> NextLayer { get; }

    /// <summary>
    /// Adds next (it terms of linked list paradigm) layer
    /// </summary>
    void AddNextLayer(IComputingLayer<TOut> layer);
  }
}
