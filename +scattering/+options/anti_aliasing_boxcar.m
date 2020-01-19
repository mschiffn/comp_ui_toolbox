%
% superclass for all boxcar spatial anti-aliasing filter options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2020-01-18
%
classdef anti_aliasing_boxcar < scattering.options.anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing_boxcar( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vectors for size
            % superclass ensures positive integers for size

            %--------------------------------------------------------------
            % 2.) create boxcar spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.anti_aliasing( size );

        end % function objects = anti_aliasing_boxcar( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( anti_aliasings_boxcar )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.anti_aliasing_boxcar
            if ~isa( anti_aliasings_boxcar, 'scattering.options.anti_aliasing_boxcar' )
                errorStruct.message = 'anti_aliasings_boxcar must be scattering.options.anti_aliasing_boxcar!';
                errorStruct.identifier = 'string:NoOptionsAntiAliasingBoxcar';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "boxcar"
            strs_out = repmat( "boxcar", size( anti_aliasings_boxcar ) );

        end % function strs_out = string( anti_aliasings_boxcar )

	end % methods

end % classdef anti_aliasing_boxcar < scattering.options.anti_aliasing
