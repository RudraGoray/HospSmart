import {describe,it,expect,vi,afterEach} from 'vitest';
import {api,query,toggleComparison,display} from './api';
afterEach(()=>vi.unstubAllGlobals());
describe('API integration helpers',()=>{
 it('encodes filters without dropping zero',()=>{expect(query({city:'New Delhi',rating:0,empty:'',missing:null})).toBe('city=New+Delhi&rating=0');});
 it('enforces a three-hospital comparison and permits removal',()=>{expect(()=>toggleComparison(['a','b','c'],'d')).toThrow('up to three');expect(toggleComparison(['a','b','c'],'a')).toEqual(['b','c']);});
 it('keeps missing values distinct from zero',()=>{expect(display(null)).toBe('Not recorded');expect(display(0)).toBe('0');});
 it('posts validated JSON to the backend route',async()=>{const fetcher=vi.fn().mockResolvedValue({ok:true,json:async()=>({prediction:4})});vi.stubGlobal('fetch',fetcher);expect(await api('/predict/rating',{beds:200})).toEqual({prediction:4});expect(fetcher).toHaveBeenCalledWith('/api/predict/rating',expect.objectContaining({method:'POST',body:'{"beds":200}'}));});
 it('surfaces structured backend failure without fake fallbacks',async()=>{vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false,json:async()=>({detail:'No dataset imported'})}));await expect(api('/recommendations',{})).rejects.toThrow('No dataset imported');});
 it('gives a useful offline error',async()=>{vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new TypeError('network')));await expect(api('/health')).rejects.toThrow('Start FastAPI');});
});
