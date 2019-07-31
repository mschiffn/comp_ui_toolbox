%
% superclass for all active spatial anti-aliasing filter options
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2019-07-29
%
classdef options_anti_aliasing_cosine < scattering.options_anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        parameter ( 1, 1 ) double { mustBePositive, mustBeLessThanOrEqual( parameter, 1 ), mustBeNonempty } = 1	% parameter

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_anti_aliasing_cosine( parameters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid parameters

            %--------------------------------------------------------------
            % 2.) create active spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options_anti_aliasing( size( parameters ) );

            % iterate active spatial anti-aliasing filter options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).parameter = parameters( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = options_anti_aliasing_cosine( parameters )

	end % methods

end % classdef options_anti_aliasing_cosine < scattering.options_anti_aliasing
