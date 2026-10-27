export interface Hospital { id:string; name:string; city:string|null; hospital_type:string|null; beds:number|null; specialties:string[]; rating:number|null; nabh:string }
export interface Filters { cities:string[]; specialties:string[]; hospital_types:string[]; nabh:string[]; available_columns:string[]; rating_range:number[]|null }
export type Json = Record<string, any>;
export async function api<T=Json>(path:string, body?:unknown, signal?:AbortSignal):Promise<T> {
  let response:Response;
  try { response = await fetch(`/api${path}`, {signal, ...(body !== undefined ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {})}); }
  catch(e) { if (e instanceof DOMException && e.name==='AbortError') throw e; throw new Error('Cannot reach the backend. Start FastAPI on port 8000, then retry.'); }
  const data = await response.json();
  if(!response.ok) throw new Error(typeof data.detail==='string' ? data.detail : 'Some inputs are invalid. Check your entries and try again.');
  return data as T;
}
export function query(values:Record<string,string|number|null|undefined>):string {
  const p = new URLSearchParams(); Object.entries(values).forEach(([k,v])=>{if(v!=='' && v!==null && v!==undefined) p.set(k,String(v));}); return p.toString();
}
export function toggleComparison(ids:string[], id:string):string[] {
  if(ids.includes(id)) return ids.filter(x=>x!==id);
  if(ids.length>=3) throw new Error('Compare up to three hospitals. Remove one first.');
  return [...ids,id];
}
export const display = (value:unknown):string => value === null || value === undefined || value === '' ? 'Not recorded' : Array.isArray(value) ? value.join(', ') || 'Not recorded' : String(value);
