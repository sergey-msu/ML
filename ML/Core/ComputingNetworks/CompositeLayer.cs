using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents composite computational layer -
  /// a hidden layer that aggregates another layers and merges its outputs
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Direct result (not final) type</typeparam>
  public abstract class CompositeLayer<TIn, TOut> : IComputingLayer<TIn>
  {
    public const int DFT_MAX_DEGREES_OF_PARALLELISM = 8;

    private ParamIdx m_ParIdx;
    private int[] m_SubLayerEndIndxs;

    private IComputingLayer<TOut> m_NextLayer;
    private IComputingLayer<TIn>[] m_SubLayers;

    public CompositeLayer()
    {
    }

    public abstract int ParamsCount { get; }
    public IComputingLayer<TOut> NextLayer { get { return m_NextLayer; } }
    public IComputingLayer<TIn>[] SubLayers { get { return m_SubLayers; } }
    public bool UseParallelism { get; set; }
    public int MaxDegreeOfParallelism { get; set; }

    public void AddNextLayer(IComputingLayer<TOut> layer)
    {
      if (layer==null)
        throw new MLException("Next layer can not be null");

      m_NextLayer = layer;
    }

    public void AddSubLayer(IComputingLayer<TIn> subLayer)
    {
      if (subLayer==null)
        throw new MLException("Sublayer can not be null");

      if (m_SubLayers==null)
        m_SubLayers = new IComputingLayer<TIn>[] { subLayer };
      else
      {
        var len = m_SubLayers.Length;
        var subLayers = new IComputingLayer<TIn>[len+1];
        for (int i=0; i<len; i++)
          subLayers[i] = m_SubLayers[i];
        subLayers[len] = subLayer;

        m_SubLayers = subLayers;
      }
    }

    public object Calculate(TIn input)
    {
      if (m_NextLayer==null)
        throw new MLException("Next layer has not been set");

      var result = DoCalculate(input);
      return m_NextLayer.Calculate(result);
    }

    protected virtual TOut DoCalculate(TIn input)
    {
      var len = m_SubLayers.Length;
      var results = new TOut[len];

      if (UseParallelism)
      {
        if (MaxDegreeOfParallelism <= 0) MaxDegreeOfParallelism = DFT_MAX_DEGREES_OF_PARALLELISM;
        var opts = new ParallelOptions { MaxDegreeOfParallelism = MaxDegreeOfParallelism };

        Parallel.For(0, len, opts, i =>
        {
          results[i] = (TOut)m_SubLayers[i].Calculate(input);
        });
      }
      else
      {
        for (int i = 0; i < len; i++)
          results[i] = (TOut)m_SubLayers[i].Calculate(input);
      }

      return MergeResults(results);
    }

    public virtual void Build(bool buildIndex = true)
    {
      if (m_NextLayer==null || m_SubLayers==null || m_SubLayers.Length <= 0)
        throw new MLException("Next layer has not been set");

      if (buildIndex) BuildIndex(0);

      var len = m_SubLayers.Length;
      for (int i=0; i<len; i++)
        m_SubLayers[i].Build(false);

      m_NextLayer.Build(false);
    }

    public int BuildIndex(int startIdx)
    {
      var selfIdx = startIdx+ParamsCount;
      var endIdx = selfIdx;

      var len = m_SubLayers.Length;
      m_SubLayerEndIndxs = new int[len];
      for (int i=0; i<len; i++)
      {
        var subEndIdx = m_SubLayers[i].BuildIndex(endIdx);
        m_SubLayerEndIndxs[i] = subEndIdx;
        endIdx = subEndIdx;
      }

      endIdx = m_NextLayer.BuildIndex(endIdx);

      m_ParIdx = new ParamIdx(startIdx, selfIdx, endIdx);
      return endIdx;
    }

    public bool TryGetParam(int idx, out double value)
    {
      if (m_ParIdx.CheckSelf(idx))
      {
        value = DoGetParam(idx-m_ParIdx.Start);
        return true;
      }

      if (m_ParIdx.CheckEnd(idx))
      {
        bool success;
        var len = m_SubLayers.Length;
        for (int i=0; i<len; i++)
        {
          var subEndIdx = m_SubLayerEndIndxs[i];
          if (idx < subEndIdx)
          {
            success = m_SubLayers[i].TryGetParam(idx, out value);
            if (!success) throw new MLCorruptedIndexException();
            return true;
          }
        }

        success = m_NextLayer.TryGetParam(idx, out value);
        if (!success) throw new MLCorruptedIndexException();
        return true;
      }

      value = 0;
      return false;
    }

    public bool TrySetParam(int idx, double value, bool isDelta)
    {
      if (m_ParIdx.CheckSelf(idx))
      {
        DoSetParam(idx-m_ParIdx.Start, value, isDelta);
        return true;
      }

      if (m_ParIdx.CheckEnd(idx))
      {
        bool success;
        var len = m_SubLayers.Length;
        for (int i=0; i<len; i++)
        {
          var subEndIdx = m_SubLayerEndIndxs[i];
          if (idx < subEndIdx)
          {
            success = m_SubLayers[i].TrySetParam(idx, value, isDelta);
            if (!success) throw new MLCorruptedIndexException();
            return true;
          }
        }

        success = m_NextLayer.TrySetParam(idx, value, isDelta);
        if (!success) throw new MLCorruptedIndexException();
        return true;
      }

      return false;
    }

    public bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars == null || pars.Length < cursor+ParamsCount)
        return false;

      DoUpdateParams(pars, isDelta, cursor);
      m_ParIdx.MoveCursor(ref cursor);

      if (m_SubLayers==null) return false;
      var len = m_SubLayers.Length;
      for (int i=0; i<len; i++)
      {
        var success = m_SubLayers[i].TryUpdateParams(pars, isDelta, ref cursor);
        if (!success) return false;
      }

      if (m_NextLayer==null) return false;

      return m_NextLayer.TryUpdateParams(pars, isDelta, ref cursor);
    }

    protected abstract TOut MergeResults(TOut[] results);
    protected abstract double DoGetParam(int idx);
    protected abstract void DoSetParam(int idx, double value, bool isDelta);
    protected abstract void DoUpdateParams(double[] pars, bool isDelta, int cursor);
  }
}
