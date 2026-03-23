// Shared constants and utilities for Bazaar

// Card styles for all good types
const cardStyles = {
    DIAMOND: { bg: 'bg-[#C8334A]', border: 'border-[#8F1F32]', borderColor: '#8F1F32', text: 'Ruby' },
    GOLD: { bg: 'bg-[#F0C24F]', border: 'border-[#B58A2E]', borderColor: '#B58A2E', text: 'Gold' },
    SILVER: { bg: 'bg-[#6FA0C8]', border: 'border-[#3F6E99]', borderColor: '#3F6E99', text: 'Silver' },
    FABRIC: { bg: 'bg-[#9A6FC4]', border: 'border-[#6B4A93]', borderColor: '#6B4A93', text: 'Fabric' },
    SPICE: { bg: 'bg-[#9ACD32]', border: 'border-[#6E9E22]', borderColor: '#6E9E22', text: 'Spice' },
    LEATHER: { bg: 'bg-[#A8744E]', border: 'border-[#6F4A2F]', borderColor: '#6F4A2F', text: 'Leather' },
    CAMEL: { bg: 'bg-[#E19A4F]', border: 'border-[#A86A2F]', borderColor: '#A86A2F', text: 'Camel' }
};

// Bonus token dot SVG patterns
const dots3SVG = 'data:image/svg+xml;base64,' + btoa(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="25" r="8" fill="white"/><circle cx="35" cy="60" r="8" fill="white"/><circle cx="65" cy="60" r="8" fill="white"/></svg>`);

const dots4SVG = 'data:image/svg+xml;base64,' + btoa(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="30" cy="30" r="8" fill="white"/><circle cx="70" cy="30" r="8" fill="white"/><circle cx="30" cy="70" r="8" fill="white"/><circle cx="70" cy="70" r="8" fill="white"/></svg>`);

const dots5SVG = 'data:image/svg+xml;base64,' + btoa(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="30" cy="30" r="8" fill="white"/><circle cx="70" cy="30" r="8" fill="white"/><circle cx="50" cy="50" r="8" fill="white"/><circle cx="30" cy="70" r="8" fill="white"/><circle cx="70" cy="70" r="8" fill="white"/></svg>`);

/**
 * Get action description from action object
 * @param {object} action - Action object with type, offered, and requested
 * @returns {string} Human-readable action description
 */
function getActionDescription(action) {
    if (!action) return '';
    
    if (action.type === 'Take') {
        const goods = Object.entries(action.requested)
            .map(([type, count]) => `${count}x ${cardStyles[type]?.text || type}`)
            .join(', ');
        return `took ${goods}`;
    } else if (action.type === 'Sell') {
        const goods = Object.entries(action.offered)
            .map(([type, count]) => `${count}x ${cardStyles[type]?.text || type}`)
            .join(', ');
        return `sold ${goods}`;
    } else if (action.type === 'Trade') {
        const offered = Object.entries(action.offered)
            .map(([type, count]) => `${count}x ${cardStyles[type]?.text || type}`)
            .join(', ');
        const requested = Object.entries(action.requested)
            .map(([type, count]) => `${count}x ${cardStyles[type]?.text || type}`)
            .join(', ');
        return `traded ${offered} for ${requested}`;
    }
    return '';
}
