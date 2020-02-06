%
% superclass for all individual signals
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2020-02-03
%
classdef signal < processing.signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal( axes, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure column vectors
            indicator_row = cellfun( @( x ) ~iscolumn( x ), samples );
            if any( indicator_row( : ) )
                errorStruct.message = 'samples must be column vectors!';
                errorStruct.identifier = 'signal:NoColumnVectors';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@processing.signal_matrix( axes, samples );

        end % function objects = signal( axes, samples )

        %------------------------------------------------------------------
        % maximum inter-signal correlation coefficients
        %------------------------------------------------------------------
        function [ xcorr_vals_max, xcorr_lags_max ] = xcorr_max( signals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.signal
            if ~isa( signals, 'processing.signal' )
                errorStruct.message = 'signals must be processing.signal!';
                errorStruct.identifier = 'xcorr_max:NoSignals';
                error( errorStruct );
            end

            % ensure identical physical units for samples
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', signals.samples );

            % ensure class math.sequence_increasing_regular
            indicator = cellfun( @( x ) ~isa( x, 'math.sequence_increasing_regular' ), { signals.axis } );
            if any( indicator( : ) )
                errorStruct.message = 'signals must have regular axes!';
                errorStruct.identifier = 'xcorr_max:NoRegularAxes';
                error( errorStruct );
            end

            % number of signals
            N_signals = numel( signals );

            % extract deltas
            deltas = cell( size( signals ) );
            for index_signal = 1:N_signals
                deltas{ index_signal } = signals( index_signal ).axis.delta;
            end

            % ensure identical deltas
            if ~isequal( deltas{ : } )
                errorStruct.message = 'Deltas of axes must be identical!';
                errorStruct.identifier = 'xcorr_max:DiverseDeltas';
                error( errorStruct );
            end

            % extract common delta
            delta_common = deltas{ 1 };

            %--------------------------------------------------------------
            % 2.) determine maximum inter-signal correlation coefficients
            %--------------------------------------------------------------
            % initialize lags w/ zeros
            xcorr_vals_max = zeros( N_signals, 1 );
            xcorr_lags_max = delta_common * zeros( N_signals, 1 );

            % iterate signals
            for index_signal = 2:N_signals

                % extract adjacent signals
                signal_act = signals( index_signal );
                signal_prev = signals( index_signal - 1 );

                % compute inter-signal correlation coefficients
                [ xcorr_vals, xcorr_lags ] = xcorr( signal_act.samples / norm( signal_act.samples ), signal_prev.samples / norm( signal_prev.samples ) );

                % detect and save maximum of cross-correlation coefficients
                [ xcorr_vals_max( index_signal ), index_max ] = max( xcorr_vals );

                % estimate relative time delays
                xcorr_lags_max( index_signal ) = xcorr_lags( index_max ) * delta_common + signal_act.axis.offset - signal_prev.axis.offset;

                % illustrate inter-element lag
%                 figure( 999 );
%                 plot( signal_act.axis.members - xcorr_lags_max( index_signal ), signal_act.samples / max( signal_act.samples ), signal_prev.axis.members, signal_prev.samples / max( signal_prev.samples ) );
%                 pause( 0.01 );

            end % for index_signal = 2:N_signals

        end % function [ xcorr_vals_max, xcorr_lags_max ] = xcorr_max( signals )

    end % methods

end % classdef signal < processing.signal_matrix
