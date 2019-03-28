%
% superclass for all mass densities
%
% author: Martin F. Schiffner
% date: 2019-03-26
% modified: 2019-03-27
%
classdef kilogram_per_cubicmeter < physical_values.mass_density

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = kilogram_per_cubicmeter( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.mass_density( varargin{ : } );

        end % function objects = kilogram_per_cubicmeter( varargin )

	end % methods

end % classdef kilogram_per_cubicmeter < physical_values.mass_density
