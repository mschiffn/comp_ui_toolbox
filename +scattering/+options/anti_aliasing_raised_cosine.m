%
% superclass for all raised-cosine spatial anti-aliasing filter options
% ( see https://en.wikipedia.org/wiki/Raised-cosine_filter )
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2019-08-03
%
classdef anti_aliasing_raised_cosine < scattering.options.anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        roll_off_factor ( 1, 1 ) double { mustBePositive, mustBeLessThanOrEqual( roll_off_factor, 1 ), mustBeNonempty } = 1 % roll-off factor

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing_raised_cosine( roll_off_factors )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid roll_off_factors

            %--------------------------------------------------------------
            % 2.) create cosine spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.anti_aliasing( size( roll_off_factors ) );

            % iterate cosine spatial anti-aliasing filter options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).roll_off_factor = roll_off_factors( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = anti_aliasing_raised_cosine( roll_off_factors )

	end % methods

end % classdef anti_aliasing_raised_cosine < scattering.options.anti_aliasing
