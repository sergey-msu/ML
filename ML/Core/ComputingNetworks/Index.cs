using System;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Index for fast search parameter within computing node
  /// </summary>
  internal struct ParamIdx
  {
    public ParamIdx(int start, int end)
    {
      if (start > end)
        throw new MLCorruptedIndexException();

      Start = start;
      End = end;
    }

    public int Start;
    public int End;

    public bool CheckEnd(int idx)
    {
      return idx < End;
    }
  }

  /// <summary>
  /// Index for fast search parameter within complex computing node
  /// </summary>
  internal struct ParamMultiIdx
  {
    private int m_Length;
    private int[] m_Thresholds;

    public ParamMultiIdx(params int[] thresholds)
    {
      if (thresholds==null || thresholds.Length<2)
        throw new MLCorruptedIndexException();

      var len = thresholds.Length;
      var prev = thresholds[0];
      m_Thresholds = new int[len];
      m_Thresholds[0] = prev;

      for (int i=1; i<len; i++)
      {
        var next = thresholds[i];
        if (prev > next) throw new MLCorruptedIndexException();
        prev = next;
        m_Thresholds[i] = next;
      }

      m_Length = len;
      m_Thresholds = thresholds;
      Start = thresholds[0];
      End = thresholds[len-1];
    }

    public int Start;
    public int End;

    public bool CheckIdx(int idx, int thIdx)
    {
      if (thIdx >= m_Length) return false;
      return idx < m_Thresholds[thIdx];
    }

    public bool CheckEnd(int idx)
    {
      return idx < End;
    }
  }

}
