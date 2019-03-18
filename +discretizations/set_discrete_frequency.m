%
% superclass for all discrete frequency sets
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-03-17
%
classdef set_discrete_frequency < discretizations.set_discrete_physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = set_discrete_frequency( input )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % set default values
            if nargin == 0
                input = physical_values.frequency( 1 );
            end

            % ensure cell array
            if ~iscell( input )
                input = { input };
            end

            % ensure class physical_values.frequency
            for index_cell = 1:numel( input )
                if ~isa( input{ index_cell }, 'physical_values.frequency' )
                    errorStruct.message     = sprintf( 'input{ %d } must be physical_values.frequency!', index_cell );
                    errorStruct.identifier	= 'set_discrete_time:NoFrequency';
                    error( errorStruct );
                end
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.set_discrete_physical_value( input );

        end % function objects = set_discrete_frequency( input )

        %------------------------------------------------------------------
        % unique values in array (overload unique function)
        %------------------------------------------------------------------
        function [ set_out, indices_unique_to_f, indices_f_to_unique ] = unique( sets_in )

            %--------------------------------------------------------------
            % 1.) numbers of discrete frequencies and cumulative sum
            %--------------------------------------------------------------
            N_samples_f = abs( sets_in(:) );
            N_samples_f_cs = [ 0; cumsum( N_samples_f ) ];

            %--------------------------------------------------------------
            % 2.) create set of unique discrete frequencies
            %--------------------------------------------------------------
            % extract unique discrete frequencies
            [ S, ia, ic ] = unique( [ sets_in.S ] );
            N_samples_f_unique = numel( S );

            % create set of unique discrete frequencies
            set_out = discretizations.set_discrete_frequency( S );

            %--------------------------------------------------------------
            % 3.) map unique frequencies to those in each set
            %--------------------------------------------------------------
            % object and frequency indices for each unique frequency
            indices_object = sum( ( repmat( ia, [ 1, numel( N_samples_f_cs ) ] ) - repmat( N_samples_f_cs(:)', [ N_samples_f_unique, 1 ] ) ) > 0, 2 );
            indices_f = ia - N_samples_f_cs( indices_object );

            % create structures with object and frequency indices for each unique frequency
            indices_unique_to_f( N_samples_f_unique ).index_object = indices_object( N_samples_f_unique );
            indices_unique_to_f( N_samples_f_unique ).index_f = indices_f( N_samples_f_unique );
            for index_f_unique = 1:(N_samples_f_unique-1)
                indices_unique_to_f( index_f_unique ).index_object = indices_object( index_f_unique );
                indices_unique_to_f( index_f_unique ).index_f = indices_f( index_f_unique );
            end

            %--------------------------------------------------------------
            % 4.) map frequencies in each set to the unique frequencies
            %--------------------------------------------------------------
            indices_f_to_unique = cell( size( sets_in ) );

            for index_set = 1:numel( sets_in )

                index_start = N_samples_f_cs( index_set ) + 1;
                index_stop = index_start + N_samples_f( index_set ) - 1;

                indices_f_to_unique{ index_set } = ic( index_start:index_stop );
            end

        end % function [ set_out, indices_unique_to_f, indices_f_to_unique ] = unique( sets_in )

	end % methods

end % classdef set_discrete_frequency < discretizations.set_discrete_physical_value
