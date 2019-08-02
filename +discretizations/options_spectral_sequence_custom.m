%
% superclass for all sequence-based spectral discretization options w/ custom time intervals
% (common frequency axis for all sequential pulse-echo measurements)
%
% author: Martin F. Schiffner
% date: 2019-08-01
% modified: 2019-08-01
%
classdef options_spectral_sequence_custom < discretizations.options_spectral_sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        interval_hull_t ( 1, 1 ) math.interval	% custom recording time interval

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spectral_sequence_custom( intervals_hull_t )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.interval
            if ~isa( intervals_hull_t, 'math.interval' )
                errorStruct.message = 'intervals_hull_t must be math.interval!';
                errorStruct.identifier = 'options_spectral_sequence_custom:NoIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.time
            auxiliary.mustBeEqualSubclasses( 'physical_values.time', intervals_hull_t.lb );

            %--------------------------------------------------------------
            % 2.) create sequence-based spectral discretization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.options_spectral_sequence( size( intervals_hull_t ) );

            % iterate sequence-based spectral discretization options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).interval_hull_t = intervals_hull_t( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = options_spectral_sequence_custom( intervals_hull_t )

    end % methods

end % classdef options_spectral_sequence_custom < discretizations.options_spectral_sequence
